"""
Benchmark: JIT vs non-JIT performance of populate().

Measures:
  - Compilation cost (first call with jit=True)
  - Warm JIT time (subsequent calls, same max_satellites → cached XLA binary)
  - Non-JIT time (baseline)
  - Speedup at several catalogue sizes

Run from the repo root:

    python benchmark/populate_jit.py

Produces:
  - A summary table printed to stdout
  - benchmark/populate_jit.png
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxhod import Zheng07, populate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_halos(n, box=500.0):
    """Generate synthetic halos drawn from a log-uniform mass distribution."""
    positions = RNG.uniform(0, box, size=(n, 3)).astype(np.float32)
    log_masses = RNG.uniform(12.0, 15.0, size=n)
    masses     = 10 ** log_masses.astype(np.float32)
    radii      = (masses / 2e14) ** (1/3) * 1.0  # rough R ∝ M^{1/3} in Mpc/h
    radii      = radii.astype(np.float32)
    return positions, masses, radii


def _time_call(fn, n_warmup=0, n_repeat=5):
    """
    Return (times_list, wall_seconds) for n_warmup + n_repeat calls.

    The warmup calls are NOT included in times_list (they are used to
    trigger JIT compilation or fill caches).
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

MODEL = Zheng07(
    log_Mmin=13.0,
    sigma_logM=0.5,
    log_M0=13.0,
    log_M1=14.0,
    alpha=1.0,
)

N_HALOS_LIST  = [50_000, 100_000, 250_000, 500_000, 1_000_000]
N_REPEAT      = 5        # warm-run repeats per configuration
MIN_MASS      = 10 ** (MODEL.log_Mmin - 1)  # keep only halos above this

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

results = []

print(f"\n{'N_halos':>10}  {'nojit_med (s)':>14}  {'compile (s)':>12}  "
      f"{'jit_med (s)':>12}  {'speedup':>8}")
print("-" * 65)

for n_halos in N_HALOS_LIST:
    pos, mass, rad = _make_halos(n_halos)

    # Determine a fixed max_satellites so both runs use the same shape.
    # (Avoids retracing due to shape changes between iterations.)
    from jaxhod.populate import _compute_max_satellites
    max_sat = _compute_max_satellites(MODEL, mass[mass >= MIN_MASS])

    key = jax.random.PRNGKey(42)

    # --- Non-JIT (baseline) ---
    def call_nojit():
        return populate(pos, mass, rad, MODEL, key,
                        max_satellites=max_sat, min_mass=MIN_MASS, jit=False)

    # One warm-up to include JAX device transfer in a fair way, but NOT
    # counting JIT compilation (there is none for non-JIT).
    call_nojit()
    nojit_times = _time_call(call_nojit, n_warmup=0, n_repeat=N_REPEAT)
    nojit_med   = np.median(nojit_times)

    # --- JIT: first call (compilation) ---
    # We reset the JIT cache so we always measure compilation fresh.
    from jaxhod.populate import _get_populate_jit
    _get_populate_jit.cache_clear()

    t0 = time.perf_counter()
    populate(pos, mass, rad, MODEL, key,
             max_satellites=max_sat, min_mass=MIN_MASS, jit=True)
    compile_time = time.perf_counter() - t0

    # --- JIT: warm calls (cached XLA binary) ---
    def call_jit():
        return populate(pos, mass, rad, MODEL, key,
                        max_satellites=max_sat, min_mass=MIN_MASS, jit=True)

    jit_times = _time_call(call_jit, n_warmup=0, n_repeat=N_REPEAT)
    jit_med   = np.median(jit_times)

    speedup = nojit_med / jit_med

    results.append({
        'n_halos':      n_halos,
        'nojit_med':    nojit_med,
        'compile_time': compile_time,
        'jit_med':      jit_med,
        'speedup':      speedup,
        'nojit_times':  nojit_times,
        'jit_times':    jit_times,
    })

    print(f"{n_halos:>10,}  {nojit_med:>14.3f}  {compile_time:>12.3f}  "
          f"{jit_med:>12.3f}  {speedup:>8.2f}x")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

n_vals       = [r['n_halos']      for r in results]
nojit_meds   = [r['nojit_med']    for r in results]
jit_meds     = [r['jit_med']      for r in results]
compile_vals = [r['compile_time'] for r in results]
speedups     = [r['speedup']      for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left panel: wall time vs n_halos ---
ax = axes[0]
ax.plot(n_vals, nojit_meds,   'o-', color='steelblue',  label='no JIT (median)')
ax.plot(n_vals, jit_meds,     's-', color='tomato',     label='JIT warm (median)')
ax.plot(n_vals, compile_vals, '^--', color='goldenrod',  label='JIT first call\n(compile + run)')
ax.set_xlabel('Number of halos')
ax.set_ylabel('Wall time (s)')
ax.set_title('populate() wall time: JIT vs no-JIT')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k'))
ax.grid(True, which='both', ls='--', alpha=0.4)

# --- Right panel: speedup vs n_halos ---
ax = axes[1]
ax.axhline(1.0, color='gray', ls='--', lw=1, label='no speedup')
ax.plot(n_vals, speedups, 'D-', color='mediumseagreen', lw=2)
ax.set_xlabel('Number of halos')
ax.set_ylabel('Speedup  (no-JIT median / JIT warm median)')
ax.set_title('Warm JIT speedup')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k'))
ax.grid(True, which='both', ls='--', alpha=0.4)
# annotate each point
for x, y in zip(n_vals, speedups):
    ax.annotate(f'{y:.1f}×', xy=(x, y), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'populate_jit.png')
fig.savefig(out_path, dpi=150)
print(f"\nFigure saved to {out_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n--- Key takeaways ---")
best_speedup = max(speedups)
best_n       = n_vals[speedups.index(best_speedup)]
avg_compile  = np.mean(compile_vals)
print(f"  Peak warm speedup : {best_speedup:.1f}× at {best_n:,} halos")
print(f"  Mean compile cost : {avg_compile:.2f} s  "
      f"(paid once; cached for subsequent calls with same max_satellites)")
breakeven_n = None
for i, (nojit, jit) in enumerate(zip(nojit_meds, jit_meds)):
    if jit < nojit:
        breakeven_n = n_vals[i]
        break
if breakeven_n:
    print(f"  JIT faster than no-JIT starting at n_halos ≥ {breakeven_n:,}")
else:
    print("  JIT warm calls are faster across all tested sizes")
