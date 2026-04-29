"""Debug: visualize zigzag detector internals on BTI7FR ~240min region."""
from __future__ import annotations
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_delta("data/flights.delta").filter(pl.col("meta_flight_id") == "4a0443_BTI7FR_s0").sort("raw_timestamp")
mach = df["bds_mach"].cast(pl.Float64).to_numpy()
ias = df["bds_ias_kt"].cast(pl.Float64).to_numpy()

# Focus on idx 3500-3700 (~240min region)
i0, i1 = 3450, 3750
t = np.arange(i0, i1)

# Compute deltas + inversion flags on BDS Mach raw
deltas = np.diff(mach)
n = len(mach)
inv_flag = np.zeros(n - 1, dtype=bool)
JUMP_MIN = 0.05
for i in range(1, n - 1):
    d_prev, d_next = deltas[i - 1], deltas[i]
    if np.isnan(d_prev) or np.isnan(d_next):
        continue
    if abs(d_prev) > JUMP_MIN and abs(d_next) > JUMP_MIN and d_prev * d_next < 0:
        inv_flag[i] = True

# Local density of inversions over half_window=15
HW = 15
density = np.zeros(n)
for i in range(n):
    lo = max(0, i - HW)
    hi = min(n - 1, i + HW)
    win = inv_flag[lo:hi]
    if win.size >= 4:
        density[i] = win.mean()

print(f"Total inversions in flight: {inv_flag.sum()}")
print(f"Inversions in idx 3500-3700: {inv_flag[3500:3700].sum()}")
print(f"Max density: {density.max():.2f} at idx {density.argmax()}")
print(f"Density in 3580-3620: max={density[3580:3620].max():.2f}, mean={density[3580:3620].mean():.2f}")

# show density values around the cluster
print("\nidx | bds_mach | inv | density")
for i in range(3580, 3620):
    inv = inv_flag[i] if i < len(inv_flag) else False
    print(f"  {i}: {mach[i]:.3f}  inv={int(inv)}  density={density[i]:.2f}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
axes[0].plot(t, mach[i0:i1], "-o", ms=3, color="tab:red", label="bds_mach raw")
axes[0].set_ylabel("Mach")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(t[:-1], deltas[i0:i1 - 1], "-o", ms=3, color="tab:purple")
axes[1].axhline(JUMP_MIN, color="k", ls="--", alpha=0.3)
axes[1].axhline(-JUMP_MIN, color="k", ls="--", alpha=0.3)
axes[1].set_ylabel("delta(mach)")
axes[1].grid(alpha=0.3)

axes[2].plot(t, density[i0:i1], "-", color="tab:blue", label="density of inversions (window=±15)")
axes[2].axhline(0.4, color="r", ls="--", alpha=0.5, label="threshold 0.4")
axes[2].set_ylabel("inv density")
axes[2].set_xlabel("idx")
axes[2].set_ylim(0, 1)
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.suptitle("Zigzag detector debug — BTI7FR Mach @ idx 3450-3750", fontsize=11)
fig.tight_layout()
fig.savefig("data/figures/debug_zigzag_BTI7FR.png", dpi=120)
print("\nwrote data/figures/debug_zigzag_BTI7FR.png")
