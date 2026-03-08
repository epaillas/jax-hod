# Getting started

## Defining an HOD model

The Zheng et al. (2007) model describes the mean number of central and satellite
galaxies as a function of halo mass using 5 parameters:

$$\langle N_\mathrm{cen}\rangle(M) = \frac{1}{2}\left[1 + \mathrm{erf}\!\left(\frac{\log_{10}M - \log_{10}M_\mathrm{min}}{\sigma_{\log M}}\right)\right]$$

$$\langle N_\mathrm{sat}\rangle(M) = \langle N_\mathrm{cen}\rangle(M)\left(\frac{M - M_0}{M_1}\right)^\alpha \quad (M > M_0)$$

```python
from jaxhod import Zheng07

model = Zheng07(
    log_Mmin=13.0,    # log10 minimum mass for a central [Msun/h]
    sigma_logM=0.5,   # width of the central occupation transition
    log_M0=13.0,      # log10 satellite cutoff mass [Msun/h]
    log_M1=14.0,      # log10 characteristic satellite mass [Msun/h]
    alpha=1.0,        # satellite power-law slope
)
```

## Populating a halo catalogue

`populate` takes generic halo arrays and returns all galaxies directly — no
padding or masking to handle:

```python
import jax
from jaxhod import populate

# Load halo data from your simulation of choice
halo_positions = ...  # (N, 3) array in Mpc/h
halo_masses    = ...  # (N,)  array in Msun/h
halo_radii     = ...  # (N,)  virial radii in Mpc/h

key = jax.random.PRNGKey(0)
result = populate(halo_positions, halo_masses, halo_radii, model, key)

gal_positions  = result['positions']   # (N_gal, 3)
gal_is_central = result['is_central']  # (N_gal,) — True for centrals
```

## Satellite radial profiles

By default satellites are placed according to an NFW density profile. The
concentration can be a scalar or a per-halo array:

```python
from jaxhod import NFW, UniformSphere

# Fixed concentration
result = populate(..., profile=NFW(concentration=5.0))

# Per-halo concentration (e.g. from a c-M relation)
concentrations = 5.71 * (halo_masses / 2e12) ** (-0.084)
result = populate(..., profile=NFW(concentration=concentrations))

# Uniform sphere (for comparison)
result = populate(..., profile=UniformSphere())
```

## Example notebook

A worked example covering all of the above, including profile comparisons and
timing benchmarks, is available in
[`nb/basic_examples.ipynb`](https://github.com/epaillas/jax-hod/blob/main/nb/basic_examples.ipynb).
