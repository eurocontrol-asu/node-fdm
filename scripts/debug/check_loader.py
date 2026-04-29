"""Quick check: resolve adsb architecture, read Delta, run dataloader."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from node_fdm_pipeline.resolver import resolve_architecture

# --- 1. Resolve architecture ---
info = resolve_architecture("adsb")
dx_col_names = [col for _, col in info.dx_cols]
all_cols = info.x_cols + info.u_cols + info.e0_cols + dx_col_names

print("=== Architecture: node_adsb_v1 ===")
print(f"  X  ({len(info.x_cols)}): {info.x_cols}")
print(f"  U  ({len(info.u_cols)}): {info.u_cols}")
print(f"  E0 ({len(info.e0_cols)}): {info.e0_cols}")
print(f"  DX ({len(dx_col_names)}): {dx_col_names}")
print(f"  Total columns needed: {len(all_cols)}")

# --- 2. Read Delta table ---
delta_path = Path("data/flights.delta")
if not delta_path.exists():
    print(f"\n[ERROR] Delta table not found at {delta_path.resolve()}")
    raise SystemExit(1)

df = pl.read_delta(str(delta_path))
print(f"\n=== Delta table: {delta_path} ===")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

# --- 3. Check required columns ---
missing = [c for c in all_cols if c not in df.columns]
present = [c for c in all_cols if c in df.columns]
print(f"\n=== Column check ===")
print(f"  Present: {len(present)}/{len(all_cols)}")
if missing:
    print(f"  [MISSING]: {missing}")
else:
    print("  All columns found!")

# --- 4. Quick stats on required columns ---
print("\n=== Column stats (non-null, NaN, min, max) ===")
for col in all_cols:
    if col not in df.columns:
        continue
    s = df[col]
    n_null = s.is_null().sum()
    n_nan = s.is_nan().sum() if s.dtype.is_float() else 0
    n_valid = len(s) - n_null - n_nan
    print(
        f"  {col:30s}  valid={n_valid:>8,}  null={n_null:>6,}  nan={n_nan:>6,}"
        f"  min={s.min():>12.4f}  max={s.max():>12.4f}"
    )

# --- 5. Run dataloader ---
print("\n=== Dataloader test ===")
df_valid = df.filter(pl.col("fdm_flag_valid"))
print(f"  Valid rows: {len(df_valid):,} / {len(df):,}")

n_flights = df_valid["meta_flight_id"].n_unique()
print(f"  Flights: {n_flights:,}")

typecodes = df_valid["meta_aircraft_type"].unique().sort().to_list()
print(f"  Typecodes: {typecodes}")

from node_fdm.loader import get_train_val_data

train_ds, val_ds = get_train_val_data(
    data_df=df_valid,
    x_cols=info.x_cols,
    u_cols=info.u_cols,
    e_cols=info.e0_cols,
    dx_cols=dx_col_names,
    seq_len=60,
    shift=60,
    train_limit=50,
    val_limit=20,
)

print(f"\n  Train samples: {len(train_ds)}")
print(f"  Val samples:   {len(val_ds)}")

if len(train_ds) > 0:
    sample = train_ds[0]
    print(f"\n  Sample shapes:")
    print(f"    x:  {tuple(sample.x.shape)}  (seq_len x {len(info.x_cols)} x_cols)")
    print(f"    u:  {tuple(sample.u.shape)}  (seq_len x {len(info.u_cols)} u_cols)")
    print(f"    e:  {tuple(sample.e.shape)}  (seq_len x {len(info.e0_cols)} e0_cols)")
    print(f"    dx: {tuple(sample.dx.shape)}  (seq_len x {len(dx_col_names)} dx_cols)")
    print(f"\n  x[0] (first timestep): {sample.x[0].tolist()}")
    print(f"  u[0] (first timestep): {sample.u[0].tolist()}")

    # Check for NaN/inf in sample
    import torch

    for name, tensor in [("x", sample.x), ("u", sample.u), ("e", sample.e), ("dx", sample.dx)]:
        finite = torch.isfinite(tensor).all().item()
        print(f"  {name} all finite: {finite}")

print("\n=== Done ===")
