# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`jax-hod` is a JAX-based Halo Occupation Distribution (HOD) framework for GPU-accelerated galaxy mock creation. All computation is JAX-native and JIT-compilable end-to-end.

## Installation

```bash
pip install -e .
# With AbacusSummit support:
pip install -e .[abacus]
# With test dependencies:
pip install -e .[test]
```

## Running Tests

```bash
pytest                          # all tests
pytest tests/test_zheng07.py    # single file
pytest tests/test_zheng07.py::TestPopulate::test_output_keys  # single test
```

## Architecture

The package exposes four top-level symbols from `jaxhod/__init__.py`:

- **`Zheng07`** (`jaxhod/models/zheng07.py`) — 5-parameter HOD model. A plain `@dataclass` with `mean_ncen(masses)` and `mean_nsat(masses)` methods returning JAX arrays.
- **`populate`** (`jaxhod/populate.py`) — Public API for galaxy population. Calls the internal `_populate()` (which returns fixed-size padded JAX arrays suitable for JIT), then filters to valid galaxies and returns NumPy arrays.
- **`NFW`, `UniformSphere`** (`jaxhod/profiles.py`) — Satellite radial profiles. Each is a `@dataclass` implementing `sample_offsets(key, n_halos, max_satellites, radii) -> (n_halos, max_satellites, 3)`. NFW uses Newton-Raphson CDF inversion; `concentration` can be a scalar or a per-halo array.
- **`load_abacus_halos`** / **`load_abacus_subsampled_halos`** (`jaxhod/simulations/abacus.py`) — Optional readers for AbacusSummit halo catalogs. Require `abacusutils`. The subsampled reader is preferred for HOD work (50–100× smaller). Returns dicts with `positions`, `masses`, `radii`, `velocities`, (`weights` for subsampled), `header`.

### Key design constraints

- **JIT compatibility**: `_populate()` uses fixed-size padded arrays (`max_satellites` slots per halo) so shapes are static. The public `populate()` filters the mask on the CPU side with NumPy to avoid shape-dependent branching in JAX.
- **HOD model interface**: Any object with `mean_ncen(masses)` and `mean_nsat(masses)` methods works as a model — not tied to `Zheng07`.
- **Profile interface**: Any object with `sample_offsets(key, n_halos, max_satellites, radii)` works as a profile.
- **Satellite assignment**: Satellites are only assigned to halos that already host a central galaxy (enforced in `_populate`).
- **Halo weights**: `halo_weights` (e.g. `multi_halos` from AbacusHOD) are post-hoc counting corrections, not per-halo occupation modifiers. They are propagated to output galaxies for number-density estimation (`n = Σweights / V`) but do NOT modify `mean_ncen`/`mean_nsat` inside `_populate`.
- **JIT caching**: `_get_populate_jit(max_satellites)` uses `functools.lru_cache` so the compiled XLA binary is reused across calls with the same `max_satellites`. Pass `max_satellites=result['max_satellites']` on repeat calls to avoid retracing.

### Benchmarks

- `benchmark/populate_jit.py` — JIT vs non-JIT wall time across 50k–1M halos. Warm JIT gives ~2× speedup on CPU; compilation cost ~0.4–1.3 s (paid once).
- `nb/memory_scaling.py` — Peak RSS vs catalogue size and batch size. ~840 MB/million halos; `batch_size=1M` → ~1.8 GB peak.
