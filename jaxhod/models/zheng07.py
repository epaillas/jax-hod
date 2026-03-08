from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.typing
from jax.scipy.special import erf


@dataclass(frozen=True)
class Zheng07:
    """
    Zheng et al. (2007) 5-parameter HOD model.

    Central galaxies follow a softened step function in halo mass, and
    satellites follow a power law above a cutoff mass.

    Parameters
    ----------
    log_Mmin : float
        Log10 of the minimum halo mass to host a central galaxy [Msun/h].
    sigma_logM : float
        Width of the central occupation transition (in dex).
    log_M0 : float
        Log10 of the satellite cutoff mass [Msun/h].
    log_M1 : float
        Log10 of the characteristic satellite mass [Msun/h].
    alpha : float
        Power-law slope of the satellite mean occupation.

    References
    ----------
    Zheng et al. 2007, ApJ, 667, 760. https://doi.org/10.1086/521074
    """

    log_Mmin: float
    sigma_logM: float
    log_M0: float
    log_M1: float
    alpha: float

    def mean_ncen(self, masses: jax.typing.ArrayLike) -> jax.Array:
        """
        Mean number of central galaxies per halo.

        .. math::

            \\langle N_\\mathrm{cen} \\rangle(M) = \\frac{1}{2}
            \\left[1 + \\mathrm{erf}\\!
            \\left(\\frac{\\log_{10}M - \\log_{10}M_\\mathrm{min}}
            {\\sigma_{\\log M}}\\right)\\right]

        Parameters
        ----------
        masses : array_like, shape (N,)
            Halo masses in Msun/h.

        Returns
        -------
        jax.Array, shape (N,)
            Mean central occupation, in [0, 1].
        """
        log_mass = jnp.log10(masses)
        return 0.5 * (1.0 + erf((log_mass - self.log_Mmin) / self.sigma_logM))

    def mean_nsat(self, masses: jax.typing.ArrayLike) -> jax.Array:
        """
        Mean number of satellite galaxies per halo.

        .. math::

            \\langle N_\\mathrm{sat} \\rangle(M) =
            \\langle N_\\mathrm{cen} \\rangle(M)
            \\left(\\frac{M - M_0}{M_1}\\right)^\\alpha
            \\quad (M > M_0)

        Parameters
        ----------
        masses : array_like, shape (N,)
            Halo masses in Msun/h.

        Returns
        -------
        jax.Array, shape (N,)
            Mean satellite occupation, >= 0.
        """
        M0 = 10.0 ** self.log_M0
        M1 = 10.0 ** self.log_M1
        ncen = self.mean_ncen(masses)
        arg = (masses - M0) / M1
        return ncen * jnp.where(arg > 0.0, jnp.power(arg, self.alpha), 0.0)
