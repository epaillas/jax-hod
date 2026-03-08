"""
Benchmark: multi-device parallel populate() vs sequential.

Compares sequential (1 device) vs parallel (N devices) batched population
for N = 1, 2, 4 where enough devices are available.  On CPU-only machines
the same physical CPU is listed multiple times to exercise the threading
path and provide a baseline; GPU runs show true parallel speedup.

Run from the repo root:

    python benchmark/multi_device.py

Produces:
  - A summary table printed to stdout
  - benchmark/multi_device.png
"""

import os
import sys
import time

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxhod import Zheng07, populate, get_devices
from jaxhod.populate import _compute_max_satellites


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = Zheng07(
    log_Mmin=13.0,
    sigma_logM=0.5,
    log_M0=13.0,
    log_M1=14.0,
    alpha=1.0,
)
MIN_MASS     = 10 ** (MODEL.log_Mmin - 1)
N_HALOS      = 500_000
BATCH_SIZE   = 100_000
N_REPEAT     = 3
RNG          = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_halos(n, box=500.0):
    pos  = RNG.uniform(0, box, size=(n, 3)).astype(np.float32)
    mass = (10 ** RNG.uniform(12.0, 15.0, size=n)).astype(np.float32)
    rad  = ((mass / 2e14) ** (1 / 3)).astype(np.float32)
    return pos, mass, rad


def _bench(pos, mass, rad, max_sat, key, devices_list, n_repeat=N_REPEAT):
    """Return median wall time (s) for populate() with the given devices list."""
    times = []
    for rep in range(n_repeat + 1):  # +1 warm-up
        t0 = time.perf_counter()
        populate(pos, mass, rad, MODEL, jax.random.fold_in(key, rep),
                 max_satellites=max_sat, min_mass=MIN_MASS,
                 batch_size=BATCH_SIZE, devices=devices_list, jit=True)
        jax.effects_barrier()
        elapsed = time.perf_counter() - t0
        if rep > 0:  # skip warm-up
            times.append(elapsed)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Discover devices
# ---------------------------------------------------------------------------

gpus = get_devices('gpu')
is_cpu_fallback = gpus[0].platform == 'cpu'

if is_cpu_fallback:
    print("NOTE: No GPU detected — using CPU devices for all slots.\n"
          "      Results show threading overhead, not true multi-GPU speedup.\n")
    # Repeat the single CPU device to exercise the parallel code path.
    physical = gpus[0]
    device_counts = [1, 2, 4]
    device_lists  = {n: [physical] * n for n in device_counts}
    label_prefix  = 'CPU'
else:
    print(f"GPUs found: {gpus}\n")
    device_counts = sorted({1, 2, min(4, len(gpus))})
    device_lists  = {n: gpus[:n] for n in device_counts}
    label_prefix  = 'GPU'

print(f"JAX version : {jax.__version__}")
print(f"N_halos     : {N_HALOS:,}")
print(f"batch_size  : {BATCH_SIZE:,}")
print(f"N_repeat    : {N_REPEAT}\n")


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

pos, mass, rad = _make_halos(N_HALOS)
keep    = mass >= MIN_MASS
max_sat = _compute_max_satellites(MODEL, mass[keep])
key     = jax.random.PRNGKey(0)

print(f"{'N devices':>10}  {'Median (s)':>11}  {'Speedup vs 1':>13}  {'vs ideal':>9}")
print("-" * 50)

rows = []
t_seq = None
for n_dev in device_counts:
    devs  = device_lists[n_dev]
    t_med = _bench(pos, mass, rad, max_sat, key, devs)
    if t_seq is None:
        t_seq = t_med
    speedup = t_seq / t_med
    ideal   = float(n_dev)
    rows.append({'n_dev': n_dev, 't_med': t_med, 'speedup': speedup, 'ideal': ideal})
    print(f"{n_dev:>10}  {t_med:>11.3f}  {speedup:>12.2f}×  "
          f"{speedup/ideal:>8.0%}")

print()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

xs      = [r['n_dev']   for r in rows]
speedup = [r['speedup'] for r in rows]
ideal   = [r['ideal']   for r in rows]

# Left: wall time
ax = axes[0]
ax.plot(xs, [r['t_med'] for r in rows], 'o-', color='steelblue',
        label=f'{label_prefix} parallel', lw=2)
ax.set_xlabel('Number of devices')
ax.set_ylabel('Median wall time (s)')
ax.set_title(f'populate() wall time\n({N_HALOS//1000}k halos, batch={BATCH_SIZE//1000}k)')
ax.set_xticks(xs)
ax.grid(True, ls='--', alpha=0.4)
ax.legend()

# Right: speedup vs ideal
ax = axes[1]
ax.plot(xs, speedup, 'o-', color='steelblue', label='actual', lw=2)
ax.plot(xs, ideal,   's--', color='gray',      label='ideal linear', lw=1.5)
for x, y in zip(xs, speedup):
    ax.annotate(f'{y:.2f}×', xy=(x, y), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9)
ax.set_xlabel('Number of devices')
ax.set_ylabel('Speedup vs 1 device')
ax.set_title('Actual vs ideal-linear speedup')
ax.set_xticks(xs)
ax.set_ylim(bottom=0)
ax.grid(True, ls='--', alpha=0.4)
ax.legend()

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'multi_device.png')
fig.savefig(out, dpi=150)
print(f"Figure saved to {out}")
