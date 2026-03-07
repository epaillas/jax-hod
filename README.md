# jax-hod

JAX-based Halo Occupation Distribution (HOD) framework for GPU-accelerated galaxy mock creation.

## Design goals

- **GPU-native**: all computation runs on JAX, JIT-compilable end-to-end.
- **Simulation-agnostic**: accepts generic halo catalogues (positions, masses, radii) — not tied to any specific simulation suite.
- **Modular**: HOD models are interchangeable objects with a minimal interface (`mean_ncen`, `mean_nsat`).

## Installation

```bash
pip install -e .
```

## Quick example

```python
import jax
import jax.numpy as jnp
from jaxhod import Zheng07, populate

# Define HOD parameters (Zheng et al. 2007)
model = Zheng07(
    log_Mmin=13.0,
    sigma_logM=0.5,
    log_M0=13.0,
    log_M1=14.0,
    alpha=1.0,
)

# Generic halo inputs (e.g. from AbacusSummit, Millennium, etc.)
halo_positions = ...  # (N, 3) array in Mpc/h
halo_masses    = ...  # (N,)  array in Msun/h
halo_radii     = ...  # (N,)  array in Mpc/h

key = jax.random.PRNGKey(0)
result = populate(halo_positions, halo_masses, halo_radii, model, key, max_satellites=50)

# result['positions']  : (N_gal, 3) — positions of all galaxies
# result['is_central'] : (N_gal,)   — True for centrals, False for satellites
gal_positions  = result['positions']
gal_is_central = result['is_central']
```

## HOD models

| Model | Parameters | Reference |
|---|---|---|
| `Zheng07` | log_Mmin, sigma_logM, log_M0, log_M1, alpha | Zheng et al. (2007) |

## Running tests

```bash
pytest
```
