from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import jax.typing
import numpy as np


@functools.lru_cache(maxsize=32)
def _get_populate_jit(max_satellites: int):
    """
    Return a JIT-compiled ``_populate`` bound to a fixed ``max_satellites``.

    Results are cached so repeated calls with the same ``max_satellites``
    return the same compiled function without retracing.

    Parameters
    ----------
    max_satellites : int
        Maximum number of satellite slots per halo.  Baked into the compiled
        function as a static constant so JAX array shapes are fixed.

    Returns
    -------
    Callable
        JIT-compiled version of ``_populate`` with ``max_satellites`` fixed.
    """
    return jax.jit(
        lambda pos, m, r, model, key, profile, hw:
            _populate(pos, m, r, model, key, max_satellites, profile, hw),
        static_argnums=(3, 5),  # model and profile are Python objects, not arrays
    )


def _compute_max_satellites(model: Any, halo_masses: np.ndarray) -> int:
    """
    Automatically determine ``max_satellites`` from the HOD model and halo masses.

    Evaluates the predicted mean satellite occupation at the most massive halo
    and adds a 5-sigma Poisson buffer so the probability of any single halo
    exceeding the cap is negligible (< 3×10⁻⁷ per halo).

    Parameters
    ----------
    model : HOD model instance
        Must expose a ``mean_nsat(masses)`` method returning a JAX array.
    halo_masses : np.ndarray, shape (N,)
        Halo masses in Msun/h.

    Returns
    -------
    int
        Suggested value for ``max_satellites``, at least 1.
    """
    max_mass = float(np.max(halo_masses))
    lam = float(model.mean_nsat(jnp.asarray([max_mass]))[0])
    if lam <= 0:
        return 1
    return max(int(np.ceil(lam + 5.0 * np.sqrt(lam))), 1)


def _populate(
    halo_positions: jax.typing.ArrayLike,
    halo_masses: jax.typing.ArrayLike,
    halo_radii: jax.typing.ArrayLike,
    model: Any,
    key: jax.Array,
    max_satellites: int = 50,
    profile: Any = None,
    halo_weights: jax.typing.ArrayLike | None = None,
) -> dict[str, jax.Array]:
    """
    JAX-native HOD population kernel.

    Returns fixed-size padded arrays plus a boolean ``mask`` so the function
    has static output shapes and is fully JIT-compatible.  Use ``populate()``
    for a cleaner interface that strips the padding and returns plain NumPy
    arrays.

    Parameters
    ----------
    halo_positions : array_like, shape (N, 3)
        Halo centre positions in Mpc/h.
    halo_masses : array_like, shape (N,)
        Halo masses in Msun/h.
    halo_radii : array_like, shape (N,)
        Halo virial radii in Mpc/h.
    model : HOD model instance
        Must expose ``mean_ncen(masses)`` and ``mean_nsat(masses)`` methods
        returning JAX arrays.
    key : jax.Array
        JAX PRNG key; split internally into three independent sub-keys for
        central draws, satellite counts, and satellite positions.
    max_satellites : int, optional
        Maximum satellite slots allocated per halo.  Satellites drawn beyond
        this cap are silently discarded.  Default is 50.
    profile : profile instance, optional
        Radial profile used to place satellites.  Must implement
        ``sample_offsets(key, n_halos, max_satellites, radii)``.
        Defaults to ``NFW(concentration=5)``.
    halo_weights : array_like, shape (N,), optional
        Per-halo importance weights propagated to output galaxies.

    Returns
    -------
    dict with keys:
        ``positions`` : jax.Array, shape (N + N*max_satellites, 3)
            Padded position array (centrals first, then satellite slots).
        ``is_central`` : jax.Array of bool, shape (N + N*max_satellites,)
            True for valid central slots.
        ``mask`` : jax.Array of bool, shape (N + N*max_satellites,)
            True for slots that hold a real galaxy.  Apply this to all
            output arrays to obtain the unpadded catalogue.
        ``weights`` : jax.Array, shape (N + N*max_satellites,)
            Only present when ``halo_weights`` is not None.  Padded array of
            per-galaxy host-halo weights.
    """
    if profile is None:
        from .profiles import NFW
        profile = NFW()

    halo_positions = jnp.asarray(halo_positions)
    halo_masses = jnp.asarray(halo_masses)
    halo_radii = jnp.asarray(halo_radii)

    n_halos = halo_masses.shape[0]
    key_cen, key_sat_occ, key_sat_pos = jax.random.split(key, 3)

    # --- Central galaxies ---
    ncen_mean = model.mean_ncen(halo_masses)                        # (N,)
    is_central = jax.random.uniform(key_cen, (n_halos,)) < ncen_mean  # (N,) bool

    # --- Satellite occupation ---
    # Satellites are only assigned to halos that already host a central.
    nsat_mean = model.mean_nsat(halo_masses)                        # (N,)
    nsat_mean = jnp.where(is_central, nsat_mean, 0.0)
    n_sat = jax.random.poisson(key_sat_occ, nsat_mean)              # (N,) int

    # --- Satellite positions ---
    offsets = profile.sample_offsets(key_sat_pos, n_halos, max_satellites, halo_radii)
    sat_positions = halo_positions[:, None, :] + offsets            # (N, max_sat, 3)

    # Mask: slot i of halo h is valid if i < n_sat[h].
    sat_mask = jnp.arange(max_satellites)[None, :] < n_sat[:, None]  # (N, max_sat)

    # --- Combine centrals and satellites ---
    sat_positions_flat = sat_positions.reshape(-1, 3)
    sat_mask_flat = sat_mask.reshape(-1)

    all_positions = jnp.concatenate([halo_positions, sat_positions_flat], axis=0)
    all_is_central = jnp.concatenate(
        [is_central, jnp.zeros(n_halos * max_satellites, dtype=bool)], axis=0
    )
    all_mask = jnp.concatenate([is_central, sat_mask_flat], axis=0)

    result: dict[str, jax.Array] = {
        'positions': all_positions,
        'is_central': all_is_central,
        'mask': all_mask,
    }

    # Each galaxy (central or satellite) inherits the weight of its host halo.
    # Centrals: one slot per halo, weight = halo_weights.
    # Satellites: max_satellites slots per halo, each repeated max_satellites times.
    if halo_weights is not None:
        halo_weights = jnp.asarray(halo_weights)
        sat_weights_flat = jnp.repeat(halo_weights, max_satellites)
        result['weights'] = jnp.concatenate([halo_weights, sat_weights_flat], axis=0)

    return result


def populate(
    halo_positions: jax.typing.ArrayLike,
    halo_masses: jax.typing.ArrayLike,
    halo_radii: jax.typing.ArrayLike,
    model: Any,
    key: jax.Array,
    max_satellites: int | None = None,
    profile: Any = None,
    min_mass: float | None = None,
    batch_size: int | None = None,
    halo_weights: jax.typing.ArrayLike | None = None,
    jit: bool = False,
) -> dict[str, Any]:
    """
    Populate dark matter halos with galaxies using the given HOD model.

    Centrals are placed at halo centres. Satellites are distributed
    according to the given radial profile (default: NFW).

    Parameters
    ----------
    halo_positions : array_like, shape (N, 3)
        Halo centre positions in Mpc/h.
    halo_masses : array_like, shape (N,)
        Halo masses in Msun/h.
    halo_radii : array_like, shape (N,)
        Halo virial radii in the same units as ``halo_positions``.
    model : HOD model instance
        Must expose ``mean_ncen(masses)`` and ``mean_nsat(masses)`` methods
        returning JAX arrays.  ``Zheng07`` satisfies this interface.
    key : jax.Array
        JAX PRNG key.
    max_satellites : int, optional
        Maximum number of satellites per halo.  Satellites beyond this
        limit are silently dropped.  When ``None`` (default), the value is
        computed automatically from ``model.mean_nsat`` evaluated at the
        most massive halo, adding a 5-sigma Poisson safety margin to make
        truncation negligible.  Set explicitly to keep JAX array shapes
        fixed across calls and avoid recompilation.
    profile : profile instance, optional
        Radial profile for satellite placement.  Must implement
        ``sample_offsets(key, n_halos, max_satellites, radii) -> array``.
        Defaults to ``NFW(concentration=5)``.
    min_mass : float, optional
        Minimum halo mass in Msun/h.  Halos below this threshold are
        discarded in NumPy *before* any JAX arrays are allocated, reducing
        peak memory.  A good default is ``10 ** (model.log_Mmin - 2)``.
    batch_size : int, optional
        Number of halos to process per batch.  When set, halos are split
        into sequential chunks of this size so peak JAX memory scales with
        ``batch_size`` rather than the full catalogue.  Each batch receives
        an independent key derived from ``key`` via ``jax.random.fold_in``.
        Results are concatenated and are statistically equivalent (though
        not bitwise-identical) to an unbatched call with the same ``key``.
    halo_weights : array_like, shape (N,), optional
        Per-halo importance weights.  When provided, each output galaxy
        inherits the weight of its host halo and a ``weights`` key is added
        to the returned dict.

        The primary use case is the AbacusHOD subsample from
        ``load_abacus_subsampled_halos``: ``prepare_sim`` stores
        ``multi_halos = 1 / p(M)`` for each kept halo.  Passing these as
        ``halo_weights`` yields a correctly normalised galaxy number density::

            nbar = result['weights'].sum() / box_volume

    jit : bool, optional
        If ``True``, JIT-compile ``_populate`` via ``jax.jit`` before each
        batch.  The compiled function is cached internally (keyed on
        ``max_satellites``), so the first call pays the compilation cost
        (~0.4–2.6 s depending on hardware) and all subsequent calls reuse
        the cached XLA binary.  On CPU, warm JIT is ~2× faster than
        no-JIT; on GPU, device parallelism dominates and the JIT benefit
        is smaller (~1.1–2.4×).  Default is ``False``.

    Returns
    -------
    dict with keys:
        ``positions`` : np.ndarray, shape (N_gal, 3)
            Positions of all galaxies (centrals and satellites).
        ``is_central`` : np.ndarray of bool, shape (N_gal,)
            True for central galaxies, False for satellites.
        ``weights`` : np.ndarray, shape (N_gal,)
            Per-galaxy host-halo importance weight.  Only present when
            ``halo_weights`` was supplied.
        ``max_satellites`` : int
            The ``max_satellites`` value used.  Store and pass back as
            ``max_satellites=result['max_satellites']`` on subsequent calls
            to keep shapes fixed and avoid JIT recompilation.

    Examples
    --------
    >>> import jax
    >>> from jaxhod import Zheng07, populate
    >>> model = Zheng07(log_Mmin=13.0, sigma_logM=0.5,
    ...                 log_M0=13.0, log_M1=14.0, alpha=1.0)
    >>> result = populate(halo_positions, halo_masses, halo_radii,
    ...                   model, jax.random.PRNGKey(0))
    >>> result['positions'].shape   # (N_gal, 3)
    >>> result['is_central'].sum()  # number of central galaxies
    """
    halo_masses = np.asarray(halo_masses)
    halo_positions = np.asarray(halo_positions)
    halo_radii = np.asarray(halo_radii)
    if halo_weights is not None:
        halo_weights = np.asarray(halo_weights)

    if min_mass is not None:
        keep = halo_masses >= min_mass
        halo_positions = halo_positions[keep]
        halo_masses = halo_masses[keep]
        halo_radii = halo_radii[keep]
        if halo_weights is not None:
            halo_weights = halo_weights[keep]

    if len(halo_masses) == 0:
        result: dict[str, Any] = {
            'positions': np.empty((0, 3), dtype=np.float32),
            'is_central': np.empty((0,), dtype=bool),
            'max_satellites': max_satellites if max_satellites is not None else 1,
        }
        if halo_weights is not None:
            result['weights'] = np.empty((0,), dtype=np.float32)
        return result

    if max_satellites is None:
        max_satellites = _compute_max_satellites(model, halo_masses)

    if batch_size is None:
        result = _populate_and_filter(halo_positions, halo_masses, halo_radii,
                                      model, key, max_satellites, profile, halo_weights,
                                      jit=jit)
        result['max_satellites'] = max_satellites
        return result

    n_halos = halo_masses.shape[0]
    all_positions = []
    all_is_central = []
    all_weights = [] if halo_weights is not None else None

    for i, start in enumerate(range(0, n_halos, batch_size)):
        end = min(start + batch_size, n_halos)
        batch_key = jax.random.fold_in(key, i)
        batch_weights = halo_weights[start:end] if halo_weights is not None else None
        chunk = _populate_and_filter(
            halo_positions[start:end],
            halo_masses[start:end],
            halo_radii[start:end],
            model, batch_key, max_satellites, profile, batch_weights,
            jit=jit,
        )
        all_positions.append(chunk['positions'])
        all_is_central.append(chunk['is_central'])
        if all_weights is not None:
            all_weights.append(chunk['weights'])

    result = {
        'positions': np.concatenate(all_positions, axis=0),
        'is_central': np.concatenate(all_is_central, axis=0),
        'max_satellites': max_satellites,
    }
    if all_weights is not None:
        result['weights'] = np.concatenate(all_weights, axis=0)
    return result


def downsample_to_nbar(
    result: dict[str, Any],
    nbar_target: float,
    box_size: float,
    key: jax.Array,
) -> dict[str, Any]:
    """
    Randomly downsample a galaxy catalogue to match a target number density.

    The true number density of the mock is estimated from ``result['weights']``
    if present (as returned by ``populate(..., halo_weights=...)``), or from
    the raw galaxy count otherwise.  Each galaxy is then kept independently
    with probability ``nbar_target / nbar_mock``.

    Parameters
    ----------
    result : dict
        Output from ``populate()``.  Must contain ``'positions'`` and
        ``'is_central'``.  If ``'weights'`` is present, it is used to
        compute the true number density; the downsampled output also
        contains ``'weights'``.
    nbar_target : float
        Target number density in (Mpc/h) :sup:`-3`, e.g. ``1e-4`` for BOSS
        CMASS-like samples.
    box_size : float
        Side length of the (cubic) simulation box in Mpc/h.
    key : jax.Array
        JAX PRNG key used to draw the keep/discard mask.

    Returns
    -------
    dict
        Same keys as ``result``, filtered to the surviving galaxies.

    Raises
    ------
    ValueError
        If ``nbar_target`` exceeds the mock number density — downsampling
        cannot increase the number of galaxies.

    Examples
    --------
    >>> result = populate(halos['positions'], halos['masses'], halos['radii'],
    ...                   model, key, halo_weights=halos['weights'])
    >>> thin = downsample_to_nbar(result, nbar_target=1e-4,
    ...                           box_size=halos['header']['BoxSize'],
    ...                           key=jax.random.PRNGKey(1))
    >>> thin['positions'].shape[0] / halos['header']['BoxSize'] ** 3
    # ≈ 1e-4
    """
    volume = box_size ** 3

    if 'weights' in result:
        nbar_mock = float(result['weights'].sum()) / volume
    else:
        nbar_mock = len(result['positions']) / volume

    f_keep = nbar_target / nbar_mock

    if f_keep > 1.0:
        raise ValueError(
            f'nbar_target ({nbar_target:.3e}) exceeds the mock number density '
            f'({nbar_mock:.3e}). Cannot upsample.'
        )

    n_gal = len(result['positions'])
    keep = np.asarray(jax.random.bernoulli(key, p=f_keep, shape=(n_gal,)))

    # Only index array-valued keys; pass scalars (e.g. max_satellites) through.
    return {k: (v[keep] if isinstance(v, np.ndarray) else v)
            for k, v in result.items()}


def _populate_and_filter(
    halo_positions: np.ndarray,
    halo_masses: np.ndarray,
    halo_radii: np.ndarray,
    model: Any,
    key: jax.Array,
    max_satellites: int,
    profile: Any,
    halo_weights: np.ndarray | None = None,
    jit: bool = False,
) -> dict[str, np.ndarray]:
    """
    Run ``_populate`` and return only valid galaxies as NumPy arrays.

    Dispatches to the JIT-cached function when ``jit=True``, strips the
    padding mask, and transfers results to the host.

    Parameters
    ----------
    halo_positions : np.ndarray, shape (N, 3)
    halo_masses : np.ndarray, shape (N,)
    halo_radii : np.ndarray, shape (N,)
    model : HOD model instance
    key : jax.Array
    max_satellites : int
    profile : profile instance or None
    halo_weights : np.ndarray, shape (N,), optional
    jit : bool, optional

    Returns
    -------
    dict with keys ``positions``, ``is_central``, and optionally ``weights``,
    each a NumPy array containing only the valid (unpadded) galaxies.
    """
    if jit:
        _populate_fn = _get_populate_jit(max_satellites)
        result = _populate_fn(halo_positions, halo_masses, halo_radii,
                              model, key, profile, halo_weights)
    else:
        result = _populate(halo_positions, halo_masses, halo_radii,
                           model, key, max_satellites, profile, halo_weights)
    mask = np.asarray(result['mask'])
    out: dict[str, np.ndarray] = {
        'positions': np.asarray(result['positions'])[mask],
        'is_central': np.asarray(result['is_central'])[mask],
    }
    if 'weights' in result:
        out['weights'] = np.asarray(result['weights'])[mask]
    return out
