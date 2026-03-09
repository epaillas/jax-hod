from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.typing
import numpy as np


def _sample_unit_vectors(
    key: jax.Array,
    shape: tuple[int, ...],
) -> jax.Array:
    """
    Sample unit vectors distributed uniformly on the unit sphere.

    Uses the standard method: draw cos(θ) ~ Uniform(-1, 1) and
    φ ~ Uniform(0, 2π), then convert to Cartesian coordinates.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    shape : tuple of int
        Shape of the output batch, e.g. ``(n_halos, max_satellites)``.
        The returned array has shape ``(*shape, 3)``.

    Returns
    -------
    jax.Array, shape (*shape, 3)
        Unit vectors with components (x, y, z).
    """
    k1, k2 = jax.random.split(key)
    cos_theta = jax.random.uniform(k1, shape, minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(k2, shape, minval=0.0, maxval=2.0 * jnp.pi)
    sin_theta = jnp.sqrt(1.0 - cos_theta ** 2)
    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = cos_theta
    return jnp.stack([x, y, z], axis=-1)


@dataclass(frozen=True)
class UniformSphere:
    """
    Place satellites uniformly within the virial sphere.

    The radial CDF is P(r) = (r/r_vir)^3, so r is sampled as
    r = r_vir * U^(1/3) where U ~ Uniform(0, 1).  Directions are
    drawn uniformly on the sphere and are independent of radii.
    """

    def sample_offsets(
        self,
        key: jax.Array,
        n_halos: int,
        max_satellites: int,
        radii: jax.typing.ArrayLike,
    ) -> jax.Array:
        """
        Sample satellite position offsets relative to halo centres.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        n_halos : int
            Number of halos.
        max_satellites : int
            Number of satellite slots per halo (fixed for JIT compatibility).
        radii : array_like, shape (n_halos,)
            Virial radius of each halo, in the same length units as the
            halo positions passed to ``populate()``.

        Returns
        -------
        jax.Array, shape (n_halos, max_satellites, 3)
            Position offsets from halo centres.
        """
        k1, k2 = jax.random.split(key)
        r = jax.random.uniform(k1, (n_halos, max_satellites)) ** (1.0 / 3.0)
        directions = _sample_unit_vectors(k2, (n_halos, max_satellites))
        return directions * r[:, :, None] * radii[:, None, None]


@dataclass(frozen=True)
class NFW:
    """
    Place satellites according to an NFW density profile.

    The NFW density profile is:

    .. math::

        \\rho(r) = \\frac{\\rho_s}{(r/r_s)(1 + r/r_s)^2}

    where :math:`r_s = r_\\mathrm{vir} / c` is the scale radius and
    :math:`c` is the concentration.  Radii are drawn by numerically
    inverting the CDF using Newton-Raphson iteration on the cumulative
    mass kernel :math:`g(x) = \\ln(1+x) - x/(1+x)`.

    Parameters
    ----------
    concentration : float or array_like, shape (N,)
        NFW concentration parameter.  Can be a scalar (same value for all
        halos) or a per-halo array of shape ``(N,)``.  Default is 5.
    n_iter : int
        Number of Newton-Raphson iterations.  The default of 10 gives
        convergence to float32 precision for any concentration in [1, 50].
    """

    concentration: jax.typing.ArrayLike = 5.0
    n_iter: int = 10

    @staticmethod
    def _g(x: jax.Array) -> jax.Array:
        """
        Cumulative NFW mass kernel.

        .. math::

            g(x) = \\ln(1 + x) - \\frac{x}{1 + x}

        Parameters
        ----------
        x : jax.Array
            Dimensionless radius r / r_s.

        Returns
        -------
        jax.Array
            g(x), same shape as ``x``.
        """
        return jnp.log1p(x) - x / (1.0 + x)

    def sample_offsets(
        self,
        key: jax.Array,
        n_halos: int,
        max_satellites: int,
        radii: jax.typing.ArrayLike,
    ) -> jax.Array:
        """
        Sample satellite position offsets relative to halo centres.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        n_halos : int
            Number of halos.
        max_satellites : int
            Number of satellite slots per halo (fixed for JIT compatibility).
        radii : array_like, shape (n_halos,)
            Virial radius of each halo, in the same length units as the
            halo positions passed to ``populate()``.

        Returns
        -------
        jax.Array, shape (n_halos, max_satellites, 3)
            Position offsets from halo centres.
        """
        k1, k2 = jax.random.split(key)

        # Broadcast concentration to (n_halos, 1) for vectorised operations.
        c = jnp.broadcast_to(
            jnp.asarray(self.concentration, dtype=jnp.float32), (n_halos,)
        )[:, None]                                            # (n_halos, 1)

        # Sample uniform random variates u in (0, 1).
        u = jax.random.uniform(k1, (n_halos, max_satellites))

        # Invert the NFW CDF: find x = r/r_s such that g(x) = u * g(c).
        # Initial guess: uniform-sphere approximation x_0 = c * u^(1/3).
        g_c = self._g(c)                                      # (n_halos, 1)
        target = u * g_c                                      # (n_halos, max_sat)
        x = c * (u ** (1.0 / 3.0))                           # (n_halos, max_sat)

        for _ in range(self.n_iter):
            g_x = self._g(x)
            # g'(x) = x / (1 + x)^2
            g_prime_x = x / (1.0 + x) ** 2
            x = x - (g_x - target) / g_prime_x
            x = jnp.clip(x, 1e-6, c)

        # x = r/r_s; r_s = r_vir/c  =>  r/r_vir = x/c
        r_over_rvir = x / c                                   # (n_halos, max_sat)

        directions = _sample_unit_vectors(k2, (n_halos, max_satellites))
        return directions * r_over_rvir[:, :, None] * radii[:, None, None]


@dataclass
class SubsampledParticles:
    """
    Place satellites at the positions of subsampled dark matter particles.

    Satellites are drawn with replacement from the per-halo particle pool,
    so each satellite position is an actual simulation particle position.
    Halos with no subsampled particles fall back to UniformSphere.

    Build with :meth:`from_flat_arrays` rather than constructing directly.

    Parameters
    ----------
    particle_offsets : np.ndarray, shape (n_halos, max_particles_per_halo, 3)
        Padded array of particle position offsets from each halo centre.
    n_particles : np.ndarray, shape (n_halos,)
        Number of valid particles per halo (values beyond this in the padded
        axis are zeros and should be ignored).
    """

    particle_offsets: np.ndarray  # (n_halos, max_particles_per_halo, 3) float32
    n_particles: np.ndarray       # (n_halos,) int32

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @classmethod
    def from_flat_arrays(
        cls,
        halo_positions: np.ndarray,
        particle_positions: np.ndarray,
        particle_halo_indices: np.ndarray,
    ) -> 'SubsampledParticles':
        """
        Organise flat particle arrays into a padded per-halo offset array.

        Parameters
        ----------
        halo_positions : np.ndarray, shape (n_halos, 3)
            Halo centre-of-mass positions.
        particle_positions : np.ndarray, shape (N_particles, 3)
            Absolute positions of all subsampled particles.
        particle_halo_indices : np.ndarray, shape (N_particles,)
            Integer index into the halos array for each particle.

        Returns
        -------
        SubsampledParticles
        """
        n_halos = len(halo_positions)

        n_particles = np.bincount(particle_halo_indices, minlength=n_halos).astype(np.int32)
        max_p = int(n_particles.max()) if n_particles.max() > 0 else 1

        offsets = np.zeros((n_halos, max_p, 3), dtype=np.float32)

        order = np.argsort(particle_halo_indices, kind='stable')
        sorted_hids = particle_halo_indices[order]
        sorted_pos  = particle_positions[order]

        # For each particle, compute the flat column index within its halo's slot
        halo_starts = np.searchsorted(sorted_hids, np.arange(n_halos))
        # local slot index: for particle at position k in sorted order,
        # its slot = k - halo_starts[sorted_hids[k]]
        flat_row = sorted_hids                                           # (N_particles,)
        flat_col = np.arange(len(sorted_hids)) - halo_starts[sorted_hids]  # (N_particles,)
        offsets[flat_row, flat_col] = sorted_pos - halo_positions[flat_row]

        return cls(
            particle_offsets=jax.device_put(offsets),
            n_particles=n_particles,
        )

    def sample_offsets(
        self,
        key: jax.Array,
        n_halos: int,
        max_satellites: int,
        radii: jax.typing.ArrayLike,
    ) -> jax.Array:
        """
        Sample satellite position offsets from the particle pool.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        n_halos : int
            Number of halos.
        max_satellites : int
            Number of satellite slots per halo (fixed for JIT compatibility).
        radii : array_like, shape (n_halos,)
            Virial radius of each halo (used only for the UniformSphere fallback).

        Returns
        -------
        jax.Array, shape (n_halos, max_satellites, 3)
            Position offsets from halo centres.
        """
        max_p = self.particle_offsets.shape[1]
        n_p = jnp.asarray(self.n_particles)          # (n_halos,)
        n_p_safe = jnp.maximum(n_p, 1)               # avoid div-by-zero in index

        floats = jax.random.uniform(key, (n_halos, max_satellites))
        idx = jnp.floor(floats * n_p_safe[:, None]).astype(jnp.int32)
        idx = jnp.clip(idx, 0, max_p - 1)

        offsets = self.particle_offsets[jnp.arange(n_halos)[:, None], idx]

        # Fallback to UniformSphere for halos with 0 particles
        fallback = UniformSphere().sample_offsets(key, n_halos, max_satellites, radii)
        has_particles = (n_p > 0)[:, None, None]     # (n_halos, 1, 1)
        return jnp.where(has_particles, offsets, fallback)
