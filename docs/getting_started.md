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

## JIT-compiled repeated calls

`populate` converts results to NumPy arrays, so it cannot be wrapped directly
in `jax.jit`. For performance-critical loops (e.g. MCMC), use the internal
`_populate` which returns fixed-size padded JAX arrays and is fully
JIT-compatible:

```python
import jax
from jaxhod.populate import _populate

populate_jit = jax.jit(
    lambda pos, m, r, k: _populate(pos, m, r, model, k, max_satellites=50)
)

# Compile once, then reuse across iterations
result = populate_jit(halo_positions, halo_masses, halo_radii, key)

# result contains padded arrays; use result['mask'] to select valid galaxies
gal_positions = result['positions'][result['mask']]
```

## Using AbacusSummit halos

`jax-hod` ships with a reader for the
[AbacusSummit](https://abacussummit.readthedocs.io) simulation suite.
It wraps `abacusutils` (install with `pip install jax-hod[abacus]`) and
returns arrays in the format expected by `populate`.

```python
import jax
from jaxhod import Zheng07, populate
from jaxhod.simulations import load_abacus_halos

# Load halos from AbacusSummit_base_c000_ph000 at z=0.5
halos = load_abacus_halos(
    sim_dir='/global/cfs/cdirs/desi/cosmosim/Abacus',
    cosmology='c000',
    phase='ph000',
    redshift=0.5,
    min_mass=1e12,    # Msun/h — skip poorly-resolved halos
)

# halos['positions']  : (N, 3) Mpc/h
# halos['masses']     : (N,)   Msun/h
# halos['radii']      : (N,)   Mpc/h  (r100 of the L2 subhalo)
# halos['velocities'] : (N, 3) km/s
# halos['header']     : dict with BoxSize, ParticleMassHMsun, etc.

# Populate with the Zheng+07 HOD model
model = Zheng07(log_Mmin=13.0, sigma_logM=0.5, log_M0=13.0, log_M1=14.0, alpha=1.0)
key = jax.random.PRNGKey(0)

result = populate(
    halos['positions'],
    halos['masses'],
    halos['radii'],
    model,
    key,
)
```

The `header` dict contains the full simulation metadata, including `BoxSize`
(Mpc/h) and `ParticleMassHMsun`, which you may need for downstream analysis.

## Example notebook

A worked example covering all of the above, including profile comparisons and
timing benchmarks, is available in
[`nb/basic_examples.ipynb`](https://github.com/epaillas/jax-hod/blob/main/nb/basic_examples.ipynb).
