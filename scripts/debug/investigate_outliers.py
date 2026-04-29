"""Investigate two reported outlier cases in cleaned BDS speeds.

Case A — BTI7FR: low IAS values (~0 kt) around 10:30 while in cruise.
Case B — ISR826: a long plateau at Mach ~0.18 / IAS ~120 kt BEFORE the flight
         that survives the cleaning pipeline.

For each case, the script prints the offending region: timestamp, altitude
(if available), bds raw, bds clean, era reference. It also computes simple
diagnostics so we can decide what filter would have caught it.
"""
from __future__ import annotations

import polars as pl
import numpy as np

DELTA = "data/flights.delta"

df = pl.read_delta(DELTA)
print("columns of interest:")
print([c for c in df.columns if any(k in c for k in ("alt", "timestamp", "mach", "ias", "tas", "flight_id"))])
print()

# ---------- Case A: BTI7FR low IAS ----------
print("=" * 80)
print("Case A — 4a0443_BTI7FR_s0 — IAS dropouts in cruise")
print("=" * 80)
a = df.filter(pl.col("meta_flight_id") == "4a0443_BTI7FR_s0").sort("raw_timestamp")
print(f"flight rows: {a.height}")
print(f"time span: {a['raw_timestamp'].min()} → {a['raw_timestamp'].max()}")

# Find IAS-clean values that look like 0-kt outliers
low_ias = a.filter(
    (pl.col("bds_ias_kt_clean") < 50)
    & pl.col("bds_ias_kt_clean").is_not_null()
)
print(f"\npoints with bds_ias_kt_clean < 50: {low_ias.height}")
if low_ias.height:
    cols = [
        "raw_timestamp",
        "bds_ias_kt",
        "bds_ias_kt_clean",
        "era_cas_kt",
        "bds_mach",
        "bds_mach_clean",
        "era_mach",
    ]
    cols = [c for c in cols if c in low_ias.columns]
    print(low_ias.select(cols).head(20))
    print("\n   (last 5)")
    print(low_ias.select(cols).tail(5))

# Compare raw vs clean: how many IAS<50 in raw vs clean?
n_raw_low = a.filter((pl.col("bds_ias_kt") < 50) & pl.col("bds_ias_kt").is_not_null()).height
n_clean_low = a.filter((pl.col("bds_ias_kt_clean") < 50) & pl.col("bds_ias_kt_clean").is_not_null()).height
print(f"\nbds_ias_kt < 50 (raw):   {n_raw_low}")
print(f"bds_ias_kt_clean < 50:   {n_clean_low}")

# ---------- Case B: ISR826 pre-flight plateau ----------
print()
print("=" * 80)
print("Case B — 738284_ISR826_s0 — pre-flight plateau (Mach ~0.18, IAS ~120)")
print("=" * 80)
b = df.filter(pl.col("meta_flight_id") == "738284_ISR826_s0").sort("raw_timestamp")
print(f"flight rows: {b.height}")
print(f"time span: {b['raw_timestamp'].min()} → {b['raw_timestamp'].max()}")

# Show first 50 valid bds_mach_clean rows
head = b.head(80)
cols = ["raw_timestamp", "bds_mach", "bds_mach_clean", "era_mach", "bds_ias_kt", "bds_ias_kt_clean", "era_cas_kt"]
cols = [c for c in cols if c in head.columns]
print("\nfirst 80 rows (cols of interest):")
with pl.Config(tbl_rows=80, tbl_cols=20):
    print(head.select(cols))

# Look at the plateau: count consecutive identical bds_mach_clean values at the start
mc = b["bds_mach_clean"].to_numpy()
print(f"\nbds_mach_clean head (first 60): {mc[:60]}")

# Distribution of bds_mach_clean unique values in first 200 rows
head_arr = mc[:200]
finite = head_arr[~np.isnan(head_arr)]
if finite.size:
    vals, counts = np.unique(np.round(finite, 4), return_counts=True)
    print("\nunique rounded bds_mach_clean values in first 200 rows (count):")
    for v, c in zip(vals, counts):
        print(f"   mach={v:.4f}  n={c}")
