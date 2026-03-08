# Performance

## GPU support

All computation in `jax-hod` is JAX-native (`jnp` operations, `jax.random`,
`jax.scipy`) and is fully GPU-compatible.  JAX dispatches to the default device
automatically, so no code changes are needed to run on a GPU — only the
environment matters.

**Using a GPU.** Wrap your call in `jax.default_device()` to target a specific
device, or set it once for the whole script:

```python
import jax

# Target a specific GPU for one call
with jax.default_device(jax.devices('gpu')[0]):
    result = populate(halo_positions, halo_masses, halo_radii, model, key,
                      jit=True)

# Or set a global default at the top of your script
jax.config.update('jax_default_device', jax.devices('gpu')[0])
```

**Host↔device transfers.** `populate()` accepts NumPy arrays and returns NumPy
arrays.  Inside `_populate`, inputs are staged to the device via
`jnp.asarray()`, and outputs are pulled back via `np.asarray()` in
`_populate_and_filter`.  These two transfers are the only host↔device
round-trips; all intermediate computation (NFW CDF inversion, random sampling,
broadcasting) runs entirely on the device.

**Verifying your device.** Check which device JAX is using at runtime:

```python
import jax
print(jax.devices())          # list all available devices
print(jax.default_backend())  # 'cpu', 'gpu', or 'tpu'
```

**Benchmarks** (`benchmark/device_comparison.py`) run automatically on all
available devices and save a comparison figure.  CPU results (laptop,
`min_mass=10^12`):

| N halos  | CPU no-JIT (s) | CPU JIT warm (s) | JIT speedup |
|---------:|---------------:|-----------------:|------------:|
|   50 000 |           0.08 |             0.04 |        1.9× |
|  100 000 |           0.13 |             0.08 |        1.8× |
|  250 000 |           0.37 |             0.19 |        2.0× |
|  500 000 |           0.80 |             0.38 |        2.1× |
|1 000 000 |           1.62 |             0.78 |        2.1× |

GPU results are not yet available from this machine; run
`python benchmark/device_comparison.py` on a GPU node to add them.

## JIT-compiled repeated calls

Pass `jit=True` to `populate()` to enable JAX JIT compilation. The compiled
function is cached internally (keyed on `max_satellites`) so the compilation
cost is paid only once, and every subsequent call reuses the cached XLA binary:

```python
key = jax.random.PRNGKey(0)

# First call: compiles + runs (~0.4–1.5 s one-time overhead)
result = populate(halo_positions, halo_masses, halo_radii, model, key,
                  jit=True)
max_sat = result['max_satellites']   # store for subsequent calls

# Subsequent calls: warm JIT (~2× faster than no-JIT on CPU)
for i in range(n_iter):
    result = populate(halo_positions, halo_masses, halo_radii, model,
                      jax.random.PRNGKey(i), max_satellites=max_sat, jit=True)
```

To avoid recompilation, fix `max_satellites` from the first result and pass it
back explicitly — if `max_satellites` changes, JAX retraces and recompiles.

For finer control (e.g. gradient-through-population), use `_populate` directly
— it returns fixed-size padded JAX arrays and is fully JIT-compatible:

```python
from jaxhod.populate import _populate

populate_jit = jax.jit(
    lambda pos, m, r, k: _populate(pos, m, r, model, k, max_satellites=50)
)

result = populate_jit(halo_positions, halo_masses, halo_radii, key)
gal_positions = result['positions'][result['mask']]
```

## Memory-efficient population of large catalogues

`_populate` allocates a `(N_halos, max_satellites, 3)` satellite-position array
before any galaxies are drawn. Measured scaling (see `nb/memory_scaling.py`)
shows peak RSS grows at roughly **840 MB per million halos** (with
`max_satellites=50`), plus a fixed ~1 GB JAX overhead. For AbacusSummit base
(399 M halos) that extrapolates to **~330 GB** without any mitigation.

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
