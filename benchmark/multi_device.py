"""
Benchmark: multi-device parallel populate() vs sequential.

Compares sequential (1 device) vs parallel (N devices) batched population.
Accepts CLI flags so it can be run as-is on Perlmutter CPU nodes (128 cores)
and GPU nodes (4 x A100).

Basic usage
-----------
    # Auto-detect backend, default sizes
    python benchmark/multi_device.py

Perlmutter CPU node (128 cores)
--------------------------------
    python benchmark/multi_device.py \\
        --backend cpu \\
        --cpu-devices 128 \\
        --device-counts 1 2 4 8 16 32 64 128 \\
        --n-halos 2_000_000 \\
        --batch-size 100_000

    # The --cpu-devices flag sets XLA_FLAGS before JAX initialises, which is
    # the only supported way to expose multiple CPU devices to JAX.

Perlmutter GPU node (4 x A100)
--------------------------------
    python benchmark/multi_device.py \\
        --backend gpu \\
        --device-counts 1 2 4 \\
        --n-halos 4_000_000 \\
        --batch-size 1_000_000

Output
------
  - Summary table to stdout
  - benchmark/multi_device.png  (or --output PATH)
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Parse args BEFORE importing JAX so we can set XLA_FLAGS in time.
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Multi-device populate() benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--backend', choices=['cpu', 'gpu', 'tpu'], default=None,
        help='JAX backend to use. Default: gpu if available, else cpu.',
    )
    p.add_argument(
        '--cpu-devices', type=int, default=1, metavar='N',
        help='Number of CPU devices to expose via XLA_FLAGS. '
             'Set to 128 on Perlmutter CPU nodes. '
             'Ignored when --backend gpu/tpu.',
    )
    p.add_argument(
        '--device-counts', type=int, nargs='+', default=None, metavar='N',
        help='Device counts to benchmark, e.g. 1 2 4. '
             'Default: all powers of 2 up to the number of available devices.',
    )
    p.add_argument(
        '--n-halos', type=lambda s: int(s.replace('_', '')),
        default=500_000, metavar='N',
        help='Total number of synthetic halos.',
    )
    p.add_argument(
        '--batch-size', type=lambda s: int(s.replace('_', '')),
        default=100_000, metavar='N',
        help='Halos per batch passed to populate().',
    )
    p.add_argument(
        '--n-repeat', type=int, default=3, metavar='N',
        help='Timing repetitions per configuration (median is reported).',
    )
    p.add_argument(
        '--output', default=None, metavar='PATH',
        help='Output figure path. Default: benchmark/multi_device.png.',
    )
    return p.parse_args()


args = _parse_args()

# Must happen before `import jax`.
if args.cpu_devices > 1:
    flag = f'--xla_force_host_platform_device_count={args.cpu_devices}'
    existing = os.environ.get('XLA_FLAGS', '')
    # Avoid duplicating the flag if the user already set it.
    if '--xla_force_host_platform_device_count' not in existing:
        os.environ['XLA_FLAGS'] = f'{existing} {flag}'.strip()

# ---------------------------------------------------------------------------
# Remaining imports (after XLA_FLAGS is set)
# ---------------------------------------------------------------------------

import time

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxhod import Zheng07, populate, get_devices
from jaxhod.populate import _compute_max_satellites


# ---------------------------------------------------------------------------
# Model / configuration
# ---------------------------------------------------------------------------

MODEL = Zheng07(
    log_Mmin=13.0,
    sigma_logM=0.5,
    log_M0=13.0,
    log_M1=14.0,
    alpha=1.0,
)
MIN_MASS = 10 ** (MODEL.log_Mmin - 1)
RNG      = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_halos(n, box=500.0):
    pos  = RNG.uniform(0, box, size=(n, 3)).astype(np.float32)
    mass = (10 ** RNG.uniform(12.0, 15.0, size=n)).astype(np.float32)
    rad  = ((mass / 2e14) ** (1 / 3)).astype(np.float32)
    return pos, mass, rad


def _bench(pos, mass, rad, max_sat, key, devices_list, batch_size, n_repeat):
    """Return median wall time (s) for populate() with the given devices list."""
    times = []
    for rep in range(n_repeat + 1):  # +1 warm-up
        t0 = time.perf_counter()
        populate(pos, mass, rad, MODEL, jax.random.fold_in(key, rep),
                 max_satellites=max_sat, min_mass=MIN_MASS,
                 batch_size=batch_size, devices=devices_list, jit=True)
        jax.effects_barrier()
        elapsed = time.perf_counter() - t0
        if rep > 0:
            times.append(elapsed)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Discover devices
# ---------------------------------------------------------------------------

backend = args.backend
if backend is None:
    # Auto-detect: prefer GPU, fall back to CPU.
    try:
        jax.devices('gpu')
        backend = 'gpu'
    except RuntimeError:
        backend = 'cpu'

all_devices = get_devices(backend)
is_cpu = all_devices[0].platform == 'cpu'

if is_cpu and args.cpu_devices > 1 and len(all_devices) < args.cpu_devices:
    print(f"WARNING: requested --cpu-devices {args.cpu_devices} but JAX only "
          f"sees {len(all_devices)} CPU devices.\n"
          f"         Ensure XLA_FLAGS was set before JAX initialised "
          f"(this script sets it automatically).\n")

print(f"JAX version  : {jax.__version__}")
print(f"Backend      : {backend}")
print(f"Devices      : {all_devices}")
if is_cpu:
    print("NOTE: CPU backend — device counts simulate thread-level parallelism,\n"
          "      not independent physical accelerators.")
print()

# ---------------------------------------------------------------------------
# Resolve device-count list
# ---------------------------------------------------------------------------

n_available = len(all_devices)

if args.device_counts is not None:
    device_counts = sorted(set(args.device_counts))
    invalid = [n for n in device_counts if n > n_available]
    if invalid:
        print(f"WARNING: requested device counts {invalid} exceed available "
              f"devices ({n_available}); they will be skipped.")
        device_counts = [n for n in device_counts if n <= n_available]
else:
    # Default: powers of 2 up to n_available, always include 1.
    device_counts = sorted({1} | {2**k for k in range(1, 10) if 2**k <= n_available})

# Build device lists: for GPU use distinct devices; for CPU repeat the same
# physical device (JAX only has one CPU device regardless of core count, but
# the ThreadPoolExecutor still exercises true thread-level parallelism).
if is_cpu:
    physical = all_devices[0]
    device_lists = {n: [physical] * n for n in device_counts}
else:
    device_lists = {n: all_devices[:n] for n in device_counts}

print(f"N_halos      : {args.n_halos:,}")
print(f"batch_size   : {args.batch_size:,}")
print(f"N_repeat     : {args.n_repeat}")
print(f"Device counts: {device_counts}")
print()

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

pos, mass, rad = _make_halos(args.n_halos)
keep    = mass >= MIN_MASS
max_sat = _compute_max_satellites(MODEL, mass[keep])
key     = jax.random.PRNGKey(0)

print(f"{'N devices':>10}  {'Median (s)':>11}  {'Speedup vs 1':>13}  {'vs ideal':>9}")
print("-" * 50)

rows   = []
t_seq  = None
label_prefix = 'CPU' if is_cpu else backend.upper()

for n_dev in device_counts:
    devs  = device_lists[n_dev]
    t_med = _bench(pos, mass, rad, max_sat, key, devs, args.batch_size, args.n_repeat)
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

xs         = [r['n_dev']             for r in rows]
t_meds     = [r['t_med']             for r in rows]
speedup    = [r['speedup']           for r in rows]
efficiency = [r['speedup'] / r['ideal'] * 100 for r in rows]

# x-axis tick labels: use log scale when the range spans more than one decade
use_log_x = (max(xs) / max(min(xs), 1)) >= 16

# Left: wall time
ax = axes[0]
ax.plot(xs, t_meds, 'o-', color='steelblue', label=f'{label_prefix} parallel', lw=2)
ax.set_xlabel('Number of devices')
ax.set_ylabel('Median wall time (s)')
ax.set_title(f'populate() wall time\n'
             f'({args.n_halos//1000}k halos, batch={args.batch_size//1000}k, {backend})')
if use_log_x:
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
else:
    ax.set_xticks(xs)
ax.grid(True, ls='--', alpha=0.4)
ax.legend()

# Right: parallel efficiency (speedup / N × 100%)
# This stays in [0, 100%] regardless of device count, avoiding the scale
# problem that crushes the actual-speedup curve when ideal reaches 128×.
ax = axes[1]
ax.plot(xs, efficiency, 'o-', color='steelblue', lw=2)
ax.axhline(100, color='gray', ls='--', lw=1.5, label='ideal (100%)')
for x, y in zip(xs, efficiency):
    va = 'bottom' if y < 90 else 'top'
    offset = 6 if va == 'bottom' else -6
    ax.annotate(f'{y:.0f}%', xy=(x, y), xytext=(0, offset),
                textcoords='offset points', ha='center', fontsize=9)
ax.set_xlabel('Number of devices')
ax.set_ylabel('Parallel efficiency  (speedup / N × 100%)')
ax.set_title('Parallel efficiency')
if use_log_x:
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
else:
    ax.set_xticks(xs)
ax.set_ylim(0, 110)
ax.grid(True, ls='--', alpha=0.4)
ax.legend()

fig.tight_layout()
out = args.output or os.path.join(os.path.dirname(__file__), 'multi_device.png')
fig.savefig(out, dpi=150)
print(f"Figure saved to {out}")
