# jax-hod

JAX-based Halo Occupation Distribution (HOD) framework for GPU-accelerated galaxy mock creation.

**Key features:**
- GPU-native: all computation runs on JAX, JIT-compilable end-to-end.
- Simulation-agnostic: accepts generic halo catalogues — not tied to any specific simulation suite.
- Modular: HOD models and radial profiles are interchangeable objects.

```{toctree}
:maxdepth: 1
:caption: User guide

installation
getting_started
abacus
performance
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```
