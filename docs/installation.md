# Installation

## Requirements

- Python ≥ 3.10
- [JAX](https://jax.readthedocs.io/en/latest/installation.html) (CPU or GPU)

## Install from source

```bash
git clone https://github.com/epaillas/jax-hod.git
cd jax-hod
pip install -e .
```

## JAX and GPU support

`jax-hod` delegates all computation to JAX. The CPU version of JAX is installed
automatically as a dependency. To enable GPU acceleration, follow the
[official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for your platform before installing `jax-hod`.
