from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


def _sample_unit_vectors(key, shape):
    """Sample unit vectors uniformly on the sphere."""
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
    r = r_vir * Uniform^(1/3).
    """

    def sample_offsets(self, key, n_halos, max_satellites, radii):
        """
        Sample satellite position offsets relative to halo centres.

        Parameters
        ----------
        key : jax.random.PRNGKey
        n_halos : int
        max_satellites : int
        radii : array, shape (n_halos,)
            Virial radii.

        Returns
        -------
        array, shape (n_halos, max_satellites, 3)
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

        rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]

    where r_s = r_vir / concentration is the scale radius. The CDF
    is inverted numerically using Newton-Raphson iteration.

    Parameters
    ----------
    concentration : float or array_like
        NFW concentration parameter. Can be a scalar (same for all
        halos) or a per-halo array of shape (N,). Default is 5.
    n_iter : int
        Number of Newton-Raphson iterations. Default of 10 gives
        convergence to float32 precision for any concentration.
    """

    concentration: float = 5.0
    n_iter: int = 10

    @staticmethod
    def _g(x):
        """Cumulative mass kernel: g(x) = ln(1+x) - x/(1+x)."""
        return jnp.log1p(x) - x / (1.0 + x)

    def sample_offsets(self, key, n_halos, max_satellites, radii):
        """
        Sample satellite position offsets relative to halo centres.

        Parameters
        ----------
        key : jax.random.PRNGKey
        n_halos : int
        max_satellites : int
        radii : array, shape (n_halos,)
            Virial radii.

        Returns
        -------
        array, shape (n_halos, max_satellites, 3)
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
