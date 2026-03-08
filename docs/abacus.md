# AbacusSummit

`jax-hod` ships two readers for the
[AbacusSummit](https://abacussummit.readthedocs.io) simulation suite. Both
require the optional `abacus` extras:

```bash
pip install jax-hod[abacus]
```

## HOD subsample (`load_abacus_subsampled_halos`) — recommended

`abacusutils` ships a script, `abacusnbody.hod.prepare_sim`, that generates a
mass-dependent probabilistic subsample of the full catalogue and writes it to
slab-wise HDF5 files. The subsample retains essentially all halos that could
plausibly host a galaxy, making it 50–100× smaller than the full catalogue.

**Step 1 — generate the subsample (once per simulation)**

```bash
python -m abacusnbody.hod.prepare_sim --path2config config/abacus_hod.yaml
```

This produces files under `{subsample_dir}/{sim_name}/z{z}/`:

```
halos_xcom_0_seed600_abacushod_oldfenv_new.h5
halos_xcom_1_seed600_abacushod_oldfenv_new.h5
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
    cosmology='c000',
    phase='ph000',
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
    halo_weights=halos['weights'],
    batch_size=1_000_000,
)
```

The `weights` array (`multi_halos` from AbacusHOD) is the inverse of each
halo's subsampling probability. High-mass halos have weight ≈ 1; lower-mass
halos may have weight > 1 because only a fraction were retained.

**Number density.** Pass weights to `populate()` and use the weighted sum to
get an unbiased estimate:

```python
box_size = halos['header']['BoxSize']
nbar = result['weights'].sum() / box_size ** 3
```

**Matching a target number density.** Use `downsample_to_nbar` to randomly thin
the catalogue to match a survey (e.g. BOSS CMASS at ~1×10⁻⁴ (Mpc/h)⁻³):

```python
from jaxhod import downsample_to_nbar

thin = downsample_to_nbar(result, nbar_target=1e-4,
                          box_size=box_size,
                          key=jax.random.PRNGKey(1))
```

For ELG/QSO tracers, pass `mt=True` to load the multi-tracer subsample:

```python
halos = load_abacus_subsampled_halos(..., mt=True)
```

## Full catalogue (`load_abacus_halos`)

For cases where you need the complete CompaSO snapshot (~400 M halos for the
base box):

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

Loading the full catalogue requires tens of GB of memory. See the
[performance guide](performance.md) for strategies to stay within workstation
memory limits.
