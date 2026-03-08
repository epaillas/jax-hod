"""
Benchmark: CPU vs GPU performance of populate().

For each available JAX device (CPU, GPU, TPU), measures:
  - No-JIT wall time (host→device transfer + compute + device→host transfer)
  - JIT warm wall time (same, but with cached XLA binary)
  - JIT compilation cost (first call)

Run from the repo root:

    python benchmark/device_comparison.py

Produces:
  - A summary table printed to stdout
  - benchmark/device_comparison.png

On CPU-only machines the script runs and produces a single-device figure
that acts as a baseline.  Run on a GPU machine to get the CPU vs GPU
comparison.

Notes on device placement
-------------------------
jax-hod's populate() takes NumPy inputs and returns NumPy outputs.  JAX
internally stages computation on the *default device*, so wrapping the call
in `jax.default_device(dev)` is sufficient to route all computation (random
sampling, NFW CDF inversion, broadcasting) to the target device.  The only
host↔device transfers are:

  - Input transfer  : jnp.asarray(np_array) inside _populate (~negligible
                      for float32 arrays on PCIe3 x16 at >10 GB/s)
  - Output transfer : np.asarray(jax_array) inside _populate_and_filter

Both are included in the measured wall time, which is the relevant figure
for typical HOD workflows where inputs are loaded from disk as NumPy.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxhod import Zheng07, populate
from jaxhod.populate import _compute_max_satellites, _get_populate_jit


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
MIN_MASS      = 10 ** (MODEL.log_Mmin - 1)
N_HALOS_LIST  = [50_000, 100_000, 250_000, 500_000, 1_000_000]
N_REPEAT      = 5
RNG           = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_halos(n, box=500.0):
    pos  = RNG.uniform(0, box, size=(n, 3)).astype(np.float32)
    mass = (10 ** RNG.uniform(12.0, 15.0, size=n)).astype(np.float32)
    rad  = ((mass / 2e14) ** (1/3)).astype(np.float32)
    return pos, mass, rad


def _bench_device(device, n_halos_list, n_repeat=N_REPEAT):
    """
    Run no-JIT and JIT benchmarks on *device* for each catalogue size.

    Returns a list of dicts with keys:
        n_halos, nojit_med, jit_compile, jit_med
    """
    rows = []
    key  = jax.random.PRNGKey(42)

    for n in n_halos_list:
        pos, mass, rad = _make_halos(n)
        keep = mass >= MIN_MASS
        max_sat = _compute_max_satellites(MODEL, mass[keep])

        with jax.default_device(device):

            # ---- no-JIT ----
            # One warm-up to load any lazy imports and trigger device init.
            populate(pos, mass, rad, MODEL, key,
                     max_satellites=max_sat, min_mass=MIN_MASS, jit=False)
            t_nojit = []
            for _ in range(n_repeat):
                t0 = time.perf_counter()
                populate(pos, mass, rad, MODEL, key,
                         max_satellites=max_sat, min_mass=MIN_MASS, jit=False)
                # block until JAX computation is complete before stopping clock
                jax.effects_barrier()
                t_nojit.append(time.perf_counter() - t0)
            nojit_med = float(np.median(t_nojit))

            # ---- JIT: compilation (first call) ----
            _get_populate_jit.cache_clear()
            t0 = time.perf_counter()
            populate(pos, mass, rad, MODEL, key,
                     max_satellites=max_sat, min_mass=MIN_MASS, jit=True)
            jax.effects_barrier()
            jit_compile = time.perf_counter() - t0

            # ---- JIT: warm calls ----
            t_jit = []
            for _ in range(n_repeat):
                t0 = time.perf_counter()
                populate(pos, mass, rad, MODEL, key,
                         max_satellites=max_sat, min_mass=MIN_MASS, jit=True)
                jax.effects_barrier()
                t_jit.append(time.perf_counter() - t0)
            jit_med = float(np.median(t_jit))

        rows.append({
            'n_halos':     n,
            'nojit_med':   nojit_med,
            'jit_compile': jit_compile,
            'jit_med':     jit_med,
            'speedup':     nojit_med / jit_med,
        })

    return rows


# ---------------------------------------------------------------------------
# Discover devices
# ---------------------------------------------------------------------------

all_devices = {}
for backend in ('cpu', 'gpu', 'tpu'):
    try:
        devs = jax.devices(backend)
        all_devices[backend] = devs[0]   # use first device of each type
    except RuntimeError:
        pass

print(f"JAX version : {jax.__version__}")
print(f"Devices found: {list(all_devices.keys())}\n")

# Warn if only CPU is available
if list(all_devices.keys()) == ['cpu']:
    print("NOTE: Only CPU detected.  Results show CPU-only baseline.")
    print("      Run on a machine with a GPU to get the CPU vs GPU comparison.\n")


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

all_results = {}   # backend_name -> list of row dicts

for backend, device in all_devices.items():
    print(f"--- Benchmarking on {device} ---")
    print(f"{'N_halos':>10}  {'no-JIT (s)':>11}  {'compile (s)':>12}  "
          f"{'JIT warm (s)':>13}  {'speedup':>8}")
    print("-" * 62)

    rows = _bench_device(device, N_HALOS_LIST)
    all_results[backend] = rows

    for r in rows:
        print(f"{r['n_halos']:>10,}  {r['nojit_med']:>11.3f}  "
              f"{r['jit_compile']:>12.3f}  {r['jit_med']:>13.3f}  "
              f"{r['speedup']:>8.2f}×")
    print()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

n_backends = len(all_results)
colors = {
    'cpu': {'nojit': '#4878CF', 'jit': '#6ACC65'},
    'gpu': {'nojit': '#D65F5F', 'jit': '#EE854A'},
    'tpu': {'nojit': '#956CB4', 'jit': '#8C613C'},
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- Left: wall time vs n_halos ----
ax = axes[0]
for backend, rows in all_results.items():
    xs      = [r['n_halos']   for r in rows]
    nojit   = [r['nojit_med'] for r in rows]
    jit_w   = [r['jit_med']   for r in rows]
    c_nojit = colors.get(backend, {}).get('nojit', 'steelblue')
    c_jit   = colors.get(backend, {}).get('jit',   'tomato')
    lbl     = backend.upper()
    ax.plot(xs, nojit, 'o-',  color=c_nojit, label=f'{lbl} no-JIT')
    ax.plot(xs, jit_w, 's--', color=c_jit,   label=f'{lbl} JIT warm')

ax.set_xlabel('Number of halos')
ax.set_ylabel('Wall time (s)')
ax.set_title('populate() wall time by device')
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k'))
ax.legend(fontsize=9)
ax.grid(True, which='both', ls='--', alpha=0.4)

# ---- Right: speedup (JIT warm / no-JIT) per device ----
ax = axes[1]
for backend, rows in all_results.items():
    xs      = [r['n_halos'] for r in rows]
    speedup = [r['speedup'] for r in rows]
    c_jit   = colors.get(backend, {}).get('jit', 'tomato')
    ax.plot(xs, speedup, 'D-', color=c_jit, label=f'{backend.upper()} JIT speedup', lw=2)
    for x, y in zip(xs, speedup):
        ax.annotate(f'{y:.1f}×', xy=(x, y), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=8)

# If both CPU and GPU available: add GPU vs no-JIT-CPU speedup bar
if 'cpu' in all_results and 'gpu' in all_results:
    cpu_rows = {r['n_halos']: r for r in all_results['cpu']}
    xs = [r['n_halos'] for r in all_results['gpu']]
    gpu_over_cpu_nojit = [
        cpu_rows[r['n_halos']]['nojit_med'] / r['jit_med']
        for r in all_results['gpu']
    ]
    ax.plot(xs, gpu_over_cpu_nojit, '^:', color='purple',
            label='GPU JIT vs CPU no-JIT', lw=2)
    for x, y in zip(xs, gpu_over_cpu_nojit):
        ax.annotate(f'{y:.1f}×', xy=(x, y), xytext=(0, -12),
                    textcoords='offset points', ha='center', fontsize=8, color='purple')

ax.axhline(1.0, color='gray', ls='--', lw=1)
ax.set_xlabel('Number of halos')
ax.set_ylabel('Speedup  (no-JIT / JIT warm, same device)')
ax.set_title('JIT warm speedup')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k'))
ax.legend(fontsize=9)
ax.grid(True, which='both', ls='--', alpha=0.4)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'device_comparison.png')
fig.savefig(out, dpi=150)
print(f"Figure saved to {out}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n--- Summary ---")
for backend, rows in all_results.items():
    best = max(rows, key=lambda r: r['speedup'])
    avg_compile = np.mean([r['jit_compile'] for r in rows])
    print(f"  {backend.upper()}: peak JIT speedup {best['speedup']:.1f}× "
          f"at {best['n_halos']:,} halos | "
          f"mean compile {avg_compile:.2f} s")

if 'cpu' in all_results and 'gpu' in all_results:
    cpu_1m = next(r for r in all_results['cpu'] if r['n_halos'] == 1_000_000)
    gpu_1m = next(r for r in all_results['gpu'] if r['n_halos'] == 1_000_000)
    ratio_nojit = cpu_1m['nojit_med'] / gpu_1m['nojit_med']
    ratio_jit   = cpu_1m['jit_med']   / gpu_1m['jit_med']
    print(f"\n  GPU vs CPU at 1M halos:")
    print(f"    no-JIT : {ratio_nojit:.1f}× faster on GPU")
    print(f"    JIT    : {ratio_jit:.1f}× faster on GPU")
