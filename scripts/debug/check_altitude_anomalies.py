"""Analyze raw altitude anomalies for flight 4009d8_BAW893_s0.

Identifies:
- Altitude jumps in climb (large discontinuities).
- Cruise spikes (low-altitude outliers in cruise).
- Compares raw_alt_ft vs fdm_alt_target_ft and flags.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

FID = "4009d8_BAW893_s0"
DELTA = Path("data/flights.delta")
OUT = Path("data/figures/altitude_anomalies_4009d8_BAW893.png")

df = pl.read_delta(str(DELTA)).filter(pl.col("meta_flight_id") == FID).sort("raw_timestamp")
print(f"Rows: {df.height}")
print(f"Columns of interest present:")
cols = [
    "raw_timestamp",
    "raw_alt_ft",
    "fdm_alt_target_ft",
    "fdm_flag_gap_altitude",
    "fdm_flag_valid",
    "fdm_flag_crop_start",
    "fdm_flag_crop_end",
]
for c in cols:
    print(f"  {c}: {c in df.columns}")

t = (df["raw_timestamp"] - df["raw_timestamp"][0]).dt.total_seconds().to_numpy() / 60.0
alt_raw = df["raw_alt_ft"].to_numpy()
alt_tgt = df["fdm_alt_target_ft"].to_numpy() if "fdm_alt_target_ft" in df.columns else None
valid = (
    df["fdm_flag_valid"].fill_null(False).to_numpy().astype(bool)
    if "fdm_flag_valid" in df.columns
    else None
)
gap_alt = (
    df["fdm_flag_gap_altitude"].fill_null(False).to_numpy().astype(bool)
    if "fdm_flag_gap_altitude" in df.columns
    else None
)

# diff per sample
dt = np.diff(t) * 60.0  # seconds
d_alt = np.diff(alt_raw)
rate_ftmin = np.where(dt > 0, d_alt / dt * 60.0, 0.0)

# stats
print("\n=== Raw altitude stats ===")
print(f"min/max:        {np.nanmin(alt_raw):.0f} / {np.nanmax(alt_raw):.0f} ft")
print(f"NaN count:      {int(np.isnan(alt_raw).sum())}")
print(f"max |Δalt| step:{np.nanmax(np.abs(d_alt)):.0f} ft  at t={t[1:][np.nanargmax(np.abs(d_alt))]:.1f} min")
print(f"max |rate|:     {np.nanmax(np.abs(rate_ftmin)):.0f} ft/min")

# detect cruise window: sustained alt > 30000 ft
cruise_mask = alt_raw > 30000
if cruise_mask.any():
    cruise_alt = alt_raw[cruise_mask]
    cruise_t = t[cruise_mask]
    median_cruise = np.nanmedian(cruise_alt)
    print(f"\n=== Cruise (alt>30kft) ===")
    print(f"median:         {median_cruise:.0f} ft")
    print(f"min in cruise:  {np.nanmin(cruise_alt):.0f} ft")
    # spikes: deviation > 2000 ft from local median
    drops = cruise_alt < median_cruise - 2000
    print(f"# samples dropping >2000ft below cruise median: {int(drops.sum())}")
    if drops.any():
        print(f"  times (min): {cruise_t[drops][:20]}")
        print(f"  values:      {cruise_alt[drops][:20]}")

# climb jumps: identify altitude steps > 1000 ft between consecutive samples
big_steps = np.where(np.abs(d_alt) > 1000)[0]
print(f"\n=== Steps > 1000 ft between consecutive samples ===")
print(f"count: {len(big_steps)}")
for idx in big_steps[:15]:
    print(
        f"  t={t[idx]:.2f}→{t[idx+1]:.2f} min  "
        f"alt {alt_raw[idx]:.0f}→{alt_raw[idx+1]:.0f}  "
        f"Δ={d_alt[idx]:+.0f} ft  dt={dt[idx]:.1f}s  "
        f"rate={rate_ftmin[idx]:+.0f} ft/min"
    )

# flag breakdown
if valid is not None:
    print(f"\n=== Flags ===")
    print(f"fdm_flag_valid: {int(valid.sum())} / {len(valid)} True")
if gap_alt is not None:
    print(f"fdm_flag_gap_altitude: {int(gap_alt.sum() if gap_alt.dtype != bool else gap_alt.sum())} True")
if "fdm_flag_crop_start" in df.columns:
    print(f"crop_start: {df['fdm_flag_crop_start'][0]}, crop_end: {df['fdm_flag_crop_end'][0]}")

# --- plot ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax = axes[0]
ax.plot(t, alt_raw, "k.-", lw=0.8, ms=2, label="raw_alt_ft")
if alt_tgt is not None:
    ax.plot(t, alt_tgt, "r--", lw=0.8, label="fdm_alt_target_ft")
if valid is not None:
    ax.scatter(t[~valid], alt_raw[~valid], c="orange", s=8, label="flag_valid=False", zorder=5)
ax.set_ylabel("Altitude [ft]")
ax.set_title(f"Raw altitude — {FID}")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t[1:], d_alt, "b.-", lw=0.6, ms=2)
ax.axhline(1000, color="r", ls="--", lw=0.5)
ax.axhline(-1000, color="r", ls="--", lw=0.5)
ax.set_ylabel("Δalt [ft / sample]")
ax.set_title("Sample-to-sample altitude change (red = ±1000 ft)")
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(t[1:], rate_ftmin, "g.-", lw=0.6, ms=2)
ax.axhline(6000, color="r", ls="--", lw=0.5, label="±6000 ft/min")
ax.axhline(-6000, color="r", ls="--", lw=0.5)
ax.set_ylabel("Vertical rate [ft/min]")
ax.set_xlabel("Time [min]")
ax.set_title("Implied vertical rate from raw altitude")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"\nSaved: {OUT}")
