from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.special import erf


@dataclass
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
    """

    log_Mmin: float
    sigma_logM: float
    log_M0: float
    log_M1: float
    alpha: float

    def mean_ncen(self, masses):
        """
        Mean number of central galaxies per halo.

        N_cen(M) = 0.5 * [1 + erf((log10(M) - log_Mmin) / sigma_logM)]

        Parameters
        ----------
        masses : array_like, shape (N,)
            Halo masses in Msun/h.

        Returns
        -------
        array, shape (N,)
            Mean central occupation, in [0, 1].
        """
        log_mass = jnp.log10(masses)
        return 0.5 * (1.0 + erf((log_mass - self.log_Mmin) / self.sigma_logM))

    def mean_nsat(self, masses):
        """
        Mean number of satellite galaxies per halo.

        N_sat(M) = N_cen(M) * ((M - M0) / M1)^alpha  for M > M0, else 0

        Parameters
        ----------
        masses : array_like, shape (N,)
            Halo masses in Msun/h.

        Returns
        -------
        array, shape (N,)
            Mean satellite occupation, >= 0.
        """
        M0 = 10.0 ** self.log_M0
        M1 = 10.0 ** self.log_M1
        ncen = self.mean_ncen(masses)
        arg = (masses - M0) / M1
        return ncen * jnp.where(arg > 0.0, jnp.power(arg, self.alpha), 0.0)
