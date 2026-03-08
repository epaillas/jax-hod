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

## Memory-efficient population of large catalogues

`_populate` allocates a `(N_halos, max_satellites, 3)` satellite-position array
before any galaxies are drawn. Measured scaling (see
`nb/memory_scaling.py`) shows peak RSS grows at roughly **840 MB per million
halos** (with `max_satellites=50`), plus a fixed ~1 GB JAX overhead. For
AbacusSummit base (399 M halos) that extrapolates to **~330 GB** without any
mitigation — far beyond any workstation.

Two options, used together, bring this down to a tractable level:

**`min_mass`** — discard halos below a mass threshold in NumPy *before* any JAX
arrays are allocated. Halos well below `log_Mmin` have negligible occupation
probability and contribute nothing but memory:

```python
# 2 dex below log_Mmin retains all halos with non-negligible occupation
result = populate(
    halos['positions'], halos['masses'], halos['radii'],
    model, key,
    min_mass=10 ** (model.log_Mmin - 2),
)
```

**`batch_size`** — process the catalogue in sequential chunks so that the JAX
peak scales with `batch_size` rather than the full catalogue. Each batch gets an
independent derived key; results are concatenated automatically. Measured peak
for a 1 M halo batch is **~1.8 GB**:

```python
result = populate(
    halos['positions'], halos['masses'], halos['radii'],
    model, key,
    min_mass=10 ** (model.log_Mmin - 2),
    batch_size=1_000_000,
)
```

Note: because each batch uses a different derived key, a batched call produces
statistically equivalent but not bitwise-identical results compared to an
unbatched call with the same `key`. Results are fully reproducible across
repeated calls given the same `key` and `batch_size`.

**Input catalogue size.** Even with batched population, the input arrays
(positions, masses, radii) for AbacusSummit base occupy ~7.4 GB as float32.
On a 15 GB machine this leaves ~5–6 GB headroom, enough for a 1 M halo batch
(~1.8 GB peak). If you also hold velocities in memory during population, free
them first:

```python
halos = load_abacus_halos(..., min_mass=10 ** (model.log_Mmin - 2))
velocities = halos.pop('velocities')   # free before population

result = populate(
    halos['positions'], halos['masses'], halos['radii'],
    model, key, batch_size=1_000_000,
)
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

`jax-hod` ships with two readers for the
[AbacusSummit](https://abacussummit.readthedocs.io) simulation suite
(both require `pip install jax-hod[abacus]`).

### Full catalogue (`load_abacus_halos`)

Reads the complete CompaSO halo catalogue from ASDF files — all
~400 M halos for the base box. Straightforward, but see the memory
discussion above before attempting this on a laptop.

### HOD subsample (`load_abacus_subsampled_halos`)  ← recommended for HOD work

`abacusutils` ships a script, `abacusnbody.hod.prepare_sim`, that
generates a mass-dependent probabilistic subsample of the full catalogue
and writes it to slab-wise HDF5 files.  The subsample retains essentially
all halos that could plausibly host a galaxy and drops the rest, making it
50–100× smaller than the full catalogue.

**Step 1 — generate the subsample (once per simulation)**

```bash
python -m abacusnbody.hod.prepare_sim --path2config config/abacus_hod.yaml
```

This produces files under `{subsample_dir}/{sim_name}/z{z}/`:

```
halos_xcom_0_seed600_abacushod_oldfenv_new.h5
halos_xcom_1_seed600_abacushod_oldfenv_new.h5
...
particles_xcom_0_seed600_abacushod_oldfenv_new.h5
...
```

**Step 2 — load and populate**

```python
import jax
from jaxhod import Zheng07, populate
from jaxhod.simulations import load_abacus_subsampled_halos

model = Zheng07(log_Mmin=13.0, sigma_logM=0.5, log_M0=13.0, log_M1=14.0, alpha=1.0)

halos = load_abacus_subsampled_halos(
    subsample_dir='/path/to/subsamples',
    sim_dir='/global/cfs/cdirs/desi/cosmosim/Abacus',
    sim_name='AbacusSummit_base_c000_ph000',
    redshift=0.5,
)

# halos['positions']  : (N_sub, 3) Mpc/h   — much smaller than the full cat
# halos['masses']     : (N_sub,)   Msun/h
# halos['radii']      : (N_sub,)   Mpc/h  (r98 of the L2 subhalo)
# halos['velocities'] : (N_sub, 3) km/s
# halos['weights']    : (N_sub,)   inverse subsampling probability
# halos['header']     : dict with BoxSize, ParticleMassHMsun, etc.

key = jax.random.PRNGKey(0)
result = populate(
    halos['positions'],
    halos['masses'],
    halos['radii'],
    model,
    key,
    batch_size=1_000_000,
)
```

The `weights` array (``multi_halos`` from AbacusHOD) is the inverse of each
halo's subsampling probability.  High-mass halos have weight ≈ 1; lower-mass
halos may have weight > 1 because only a fraction were retained.  For number-
density estimates you should weight by this field.

For ELG/QSO tracers, pass ``mt=True`` to load the multi-tracer subsample:

```python
halos = load_abacus_subsampled_halos(..., mt=True)
```

### Full catalogue (`load_abacus_halos`)

For cases where you need the complete snapshot:

```python
from jaxhod.simulations import load_abacus_halos

halos = load_abacus_halos(
    sim_dir='/global/cfs/cdirs/desi/cosmosim/Abacus',
    cosmology='c000',
    phase='ph000',
    redshift=0.5,
    min_mass=1e12,
)
```

See the memory section above for caveats about loading the full ~400 M halo
catalogue on a workstation.

## Example notebook

A worked example covering all of the above, including profile comparisons and
timing benchmarks, is available in
[`nb/basic_examples.ipynb`](https://github.com/epaillas/jax-hod/blob/main/nb/basic_examples.ipynb).
