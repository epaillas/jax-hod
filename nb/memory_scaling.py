"""
Memory scaling benchmark for jaxhod.populate.

Measures peak RSS memory as a function of halo-catalogue size and batch_size,
fits a linear model, and extrapolates to AbacusSummit scale.

Run from the repo root::

    python nb/memory_scaling.py

Outputs ``nb/memory_scaling.png``.
"""

import subprocess
import sys
import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Worker snippets — each measurement runs in a fresh subprocess so that JAX
# compilation caches and XLA runtime state do not accumulate between points.
# ---------------------------------------------------------------------------

_WORKER_POPULATE = textwrap.dedent("""\
    import sys, os, gc, threading, time
    import numpy as np
    import jax
    import psutil
    from jaxhod import Zheng07
    from jaxhod.populate import _populate

    n_halos  = int(sys.argv[1])
    max_sat  = int(sys.argv[2])
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    model = Zheng07(log_Mmin=13.0, sigma_logM=0.5, log_M0=13.0, log_M1=14.0, alpha=1.0)
    key   = jax.random.PRNGKey(0)
    rng   = np.random.default_rng(42)

    wm = (10**rng.uniform(12,15.5,500)).astype(np.float32)
    wp = rng.uniform(0,1000,(500,3)).astype(np.float32)
    wr = (wm/1e14)**(1/3)
    r  = _populate(wp, wm, wr, model, key, max_satellites=max_sat)
    r['positions'].block_until_ready()
    del r, wm, wp, wr; gc.collect()

    masses = (10**rng.uniform(12,15.5,n_halos)).astype(np.float32)
    pos    = rng.uniform(0,1000,(n_halos,3)).astype(np.float32)
    radii  = (masses/1e14)**(1/3)

    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = [False]

    def sample():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    r = _populate(pos, masses, radii, model, key, max_satellites=max_sat)
    r['positions'].block_until_ready()
    stop[0] = True; t.join()

    print((peak[0] - baseline) / 1024**2)
""")

_WORKER_BATCH = textwrap.dedent("""\
    import sys, os, gc, threading, time
    import numpy as np
    import jax
    import psutil
    from jaxhod import Zheng07, populate

    n_halos    = int(sys.argv[1])
    batch_size = None if sys.argv[2] == 'none' else int(sys.argv[2])
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    model = Zheng07(log_Mmin=13.0, sigma_logM=0.5, log_M0=13.0, log_M1=14.0, alpha=1.0)
    key   = jax.random.PRNGKey(0)
    rng   = np.random.default_rng(42)

    wm = (10**rng.uniform(12,15.5,500)).astype(np.float32)
    wp = rng.uniform(0,1000,(500,3)).astype(np.float32)
    wr = (wm/1e14)**(1/3)
    r  = populate(wp, wm, wr, model, key)
    del r, wm, wp, wr; gc.collect()

    masses = (10**rng.uniform(12,15.5,n_halos)).astype(np.float32)
    pos    = rng.uniform(0,1000,(n_halos,3)).astype(np.float32)
    radii  = (masses/1e14)**(1/3)

    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = [False]

    def sample():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    populate(pos, masses, radii, model, key, batch_size=batch_size)
    stop[0] = True; t.join()

    print((peak[0] - baseline) / 1024**2)
""")


def _run_worker(code: str, args: list[str]) -> float:
    """Run a worker snippet in a fresh interpreter and return the printed float."""
    cmd = [sys.executable, "-c", code] + [str(a) for a in args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# Benchmark 1: peak memory vs n_halos  (no batching, max_sat=50)
# ---------------------------------------------------------------------------

N_HALOS_POINTS = [25_000, 50_000, 100_000, 250_000, 500_000,
                  1_000_000, 2_000_000, 3_000_000]
MAX_SAT = 50

print("Benchmarking peak memory vs n_halos (max_satellites=50) ...")
peak_vs_n = []
for n in N_HALOS_POINTS:
    mb = _run_worker(_WORKER_POPULATE, [n, MAX_SAT])
    peak_vs_n.append(mb)
    print(f"  n={n:>9,}  peak Δ = {mb:6.0f} MB")

# ---------------------------------------------------------------------------
# Benchmark 2: peak memory vs batch_size  (3 M halos, max_sat=50)
# ---------------------------------------------------------------------------

N_FIXED        = 3_000_000
BATCH_POINTS   = [50_000, 100_000, 250_000, 500_000, 1_000_000, 3_000_000]

print(f"\nBenchmarking peak memory vs batch_size (n_halos={N_FIXED:,}) ...")
peak_vs_batch = []
for bs in BATCH_POINTS:
    mb = _run_worker(_WORKER_BATCH, [N_FIXED, bs])
    peak_vs_batch.append(mb)
    print(f"  batch_size={bs:>9,}  peak Δ = {mb:6.0f} MB")

# Also measure unbatched
mb_unbatched = _run_worker(_WORKER_BATCH, [N_FIXED, "none"])
print(f"  batch_size=   unbatched  peak Δ = {mb_unbatched:6.0f} MB")

# ---------------------------------------------------------------------------
# Linear fit to n_halos data (use the four largest points to capture
# the asymptotic slope, away from fixed JAX startup overhead)
# ---------------------------------------------------------------------------

n_arr   = np.array(N_HALOS_POINTS, dtype=float)
mem_arr = np.array(peak_vs_n)

fit_mask = n_arr >= 500_000
slope, intercept = np.polyfit(n_arr[fit_mask], mem_arr[fit_mask], 1)
bytes_per_halo = slope * 1024**2 / 1  # MB → bytes per halo

print(f"\nLinear fit (n ≥ 500k):  slope = {slope*1e6:.2f} MB/million halos"
      f"  ({bytes_per_halo:.0f} bytes/halo)")
print(f"Fixed overhead (intercept): {intercept:.0f} MB")

# ---------------------------------------------------------------------------
# Extrapolate to AbacusSummit
# ---------------------------------------------------------------------------

N_ABACUS     = 398_851_371        # total halos in AbacusSummit base
CATALOG_MB   = N_ABACUS * (3 + 1 + 1) * 4 / 1024**2   # pos+mass+radii float32
MACHINE_MB   = 15 * 1024           # 15 GB

peak_abacus_unbatched  = intercept + slope * N_ABACUS
peak_abacus_batch_1M   = intercept + slope * 1_000_000

print(f"\nExtrapolation to AbacusSummit ({N_ABACUS:,} halos):")
print(f"  Input catalog alone (pos+mass+radii, float32): {CATALOG_MB/1024:.1f} GB")
print(f"  _populate peak without batching:  {peak_abacus_unbatched/1024:.0f} GB")
print(f"  _populate peak with batch_size=1M: {peak_abacus_batch_1M/1024:.1f} GB")
print(f"  Machine RAM: {MACHINE_MB/1024:.0f} GB")
print(f"  Catalog + batch_1M peak: {(CATALOG_MB + peak_abacus_batch_1M)/1024:.1f} GB")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.35)

# --- Panel 1: peak memory vs n_halos ---
ax = axes[0]

ax.scatter(n_arr / 1e6, mem_arr / 1024, color="steelblue", zorder=3,
           label="Measured (max_sat=50)")

# Fitted line over measured range + extrapolation to 10 M
n_extrap = np.linspace(0, 10e6, 300)
mem_fit  = intercept + slope * n_extrap
ax.plot(n_extrap / 1e6, mem_fit / 1024, "--", color="steelblue",
        alpha=0.7, label=f"Linear fit\n({slope*1e6:.0f} MB / 10⁶ halos)")

# Machine RAM limit
ax.axhline(MACHINE_MB / 1024, color="crimson", lw=1.5, ls=":",
           label=f"Machine RAM ({MACHINE_MB/1024:.0f} GB)")

# Shade the OOM region
ax.fill_between([0, 10], MACHINE_MB / 1024, ax.get_ylim()[1] if ax.get_ylim()[1] > MACHINE_MB/1024 else MACHINE_MB/1024 * 1.3,
                color="crimson", alpha=0.07)

ax.set_xlabel("Halo catalogue size  [millions]")
ax.set_ylabel("Peak RSS increase  [GB]")
ax.set_title("Memory scaling: catalogue size\n(no batching, max_satellites=50)")
ax.set_xlim(0, 10)
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax.legend(fontsize=8)

# Annotation: AbacusSummit is off-chart
ax.annotate(
    f"AbacusSummit base\n{N_ABACUS/1e6:.0f}M halos\n"
    f"→ {peak_abacus_unbatched/1024:.0f} GB (extrapolated)",
    xy=(10, mem_fit[-1] / 1024), xycoords="data",
    xytext=(6, MACHINE_MB / 1024 * 0.5),
    fontsize=7.5, color="navy",
    arrowprops=dict(arrowstyle="->", color="navy", lw=0.8),
)

# --- Panel 2: peak memory vs batch_size ---
ax2 = axes[1]

bs_arr  = np.array(BATCH_POINTS, dtype=float)
bm_arr  = np.array(peak_vs_batch)

ax2.scatter(bs_arr / 1e6, bm_arr / 1024, color="darkorange", zorder=3,
            label=f"Measured (n_halos={N_FIXED//1e6:.0f}M)")
ax2.axhline(mb_unbatched / 1024, color="gray", lw=1.5, ls="--",
            label=f"No batching ({mb_unbatched:.0f} MB)")
ax2.axhline(MACHINE_MB / 1024, color="crimson", lw=1.5, ls=":",
            label=f"Machine RAM ({MACHINE_MB/1024:.0f} GB)")

# Linear fit through batch points
slope_b, intercept_b = np.polyfit(bs_arr, bm_arr, 1)
bs_line = np.linspace(0, max(bs_arr) * 1.05, 200)
ax2.plot(bs_line / 1e6, (intercept_b + slope_b * bs_line) / 1024,
         "--", color="darkorange", alpha=0.7,
         label=f"Linear fit\n({slope_b*1e6:.0f} MB / 10⁶ batch)")

ax2.set_xlabel("batch_size  [millions]")
ax2.set_ylabel("Peak RSS increase  [GB]")
ax2.set_title(f"Memory scaling: batch size\n(n_halos={N_FIXED//1_000_000}M, max_satellites=50)")
ax2.set_xlim(0, None)
ax2.set_ylim(0, None)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax2.legend(fontsize=8)

out_path = os.path.join(os.path.dirname(__file__), "memory_scaling.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
plt.show()
