# Performance

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

## JIT-compiled repeated calls

Pass `jit=True` to `populate()` to enable JAX JIT compilation. The compiled
function is cached internally (keyed on `max_satellites`) so the compilation
cost is paid only once, and every subsequent call reuses the cached XLA binary:

```python
key = jax.random.PRNGKey(0)

# First call: compiles + runs (~0.4–1.3 s one-time overhead)
result = populate(halo_positions, halo_masses, halo_radii, model, key,
                  jit=True)
max_sat = result['max_satellites']   # store for subsequent calls

# Subsequent calls: warm JIT (2–2.4× faster than non-JIT)
for i in range(n_iter):
    result = populate(halo_positions, halo_masses, halo_radii, model,
                      jax.random.PRNGKey(i), max_satellites=max_sat, jit=True)
```

To avoid recompilation, fix `max_satellites` from the first result and pass it
back explicitly on all subsequent calls — if `max_satellites` changes, JAX
retraces and recompiles.

Benchmarks on a laptop CPU (`benchmark/populate_jit.py`) measured across
50k–1M halos (Zheng07, `min_mass=10^12`):

| N halos  | no-JIT (s) | JIT warm (s) | speedup | compile (s) |
|---------:|-----------:|-------------:|--------:|------------:|
|   50 000 |       0.08 |         0.04 |   2.0×  |        0.45 |
|  100 000 |       0.14 |         0.08 |   1.8×  |        0.39 |
|  250 000 |       0.45 |         0.19 |   2.4×  |        0.59 |
|  500 000 |       0.79 |         0.38 |   2.1×  |        0.79 |
|1 000 000 |       1.64 |         0.76 |   2.2×  |        1.34 |

The warm speedup is consistently **~2×** on CPU. Larger gains are expected on
GPU where XLA kernels are more efficiently fused. The compilation overhead
(~0.4–1.3 s) is amortised after only 1–2 warm calls, so `jit=True` is
worthwhile whenever `populate()` is called more than once (e.g. MCMC chains,
parameter sweeps).

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
