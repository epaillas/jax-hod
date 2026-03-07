import jax
import jax.numpy as jnp
import numpy as np


def _populate(halo_positions, halo_masses, halo_radii, model, key, max_satellites=50, profile=None):
    """
    Internal JAX-native population function.

    Returns fixed-size padded arrays with a boolean mask so the function
    is fully JIT-compatible. Use ``populate()`` for a cleaner interface
    that returns only valid galaxies.
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

    return {
        'positions': all_positions,
        'is_central': all_is_central,
        'mask': all_mask,
    }


def populate(halo_positions, halo_masses, halo_radii, model, key, max_satellites=50, profile=None):
    """
    Populate dark matter halos with galaxies using the given HOD model.

    Centrals are placed at halo centres. Satellites are distributed
    according to the given radial profile (default: NFW).

    Parameters
    ----------
    halo_positions : array_like, shape (N, 3)
        Halo centre positions.
    halo_masses : array_like, shape (N,)
        Halo masses in Msun/h.
    halo_radii : array_like, shape (N,)
        Halo virial radii in the same units as halo_positions.
    model : HOD model instance
        Must expose ``mean_ncen(masses)`` and ``mean_nsat(masses)`` methods.
    key : jax.random.PRNGKey
        JAX random key.
    max_satellites : int
        Maximum number of satellites per halo. Satellites beyond this
        limit are silently dropped; increase if your most massive halos
        are expected to host more than this many satellites.
    profile : profile instance, optional
        Radial profile for satellite placement. Must implement
        ``sample_offsets(key, n_halos, max_satellites, radii)``.
        Defaults to ``NFW(concentration=5)``.

    Returns
    -------
    dict with keys:
        ``positions`` : array, shape (N_gal, 3)
            Positions of all galaxies.
        ``is_central`` : bool array, shape (N_gal,)
            True for central galaxies, False for satellites.
    """
    result = _populate(
        halo_positions, halo_masses, halo_radii, model, key, max_satellites, profile
    )
    mask = np.asarray(result['mask'])
    return {
        'positions': np.asarray(result['positions'])[mask],
        'is_central': np.asarray(result['is_central'])[mask],
    }
