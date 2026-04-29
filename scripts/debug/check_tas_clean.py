"""Compare TAS sources: ERA5 vs BDS raw vs BDS cleaned.

Picks a few flights with non-trivial bds_tas_kt coverage and plots
the three signals on the same axis.

Usage:
    uv run python scripts/check_tas_clean.py [--n N] [--seed SEED]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from deltalake import DeltaTable

# bds_tas_kt is actually in knots — verified by sampling raw values
# (subagent claim about m/s mislabel was wrong; comment in clean_speeds.py:10 is misleading)


def plot_flight(df: pl.DataFrame, out_path: Path) -> None:
    n = len(df)
    t = np.arange(n) * 4 / 60  # minutes (4 s sampling)
    fid = df["meta_flight_id"][0]

    era_tas = df["era_tas_kt"].to_numpy() if "era_tas_kt" in df.columns else np.full(n, np.nan)
    bds_tas_raw = df["bds_tas_kt"].to_numpy() if "bds_tas_kt" in df.columns else np.full(n, np.nan)
    bds_tas_clean = (
        df["bds_tas_kt_clean"].to_numpy()
        if "bds_tas_kt_clean" in df.columns
        else np.full(n, np.nan)
    )

    # bds_tas is already in knots — no conversion needed
    bds_tas_raw_kt = bds_tas_raw
    bds_tas_clean_kt = bds_tas_clean

    n_raw = (~np.isnan(bds_tas_raw)).sum()
    n_clean = (~np.isnan(bds_tas_clean)).sum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, era_tas, color="#3498db", linewidth=0.9, alpha=0.8, label=f"ERA5 TAS ({(~np.isnan(era_tas)).sum()} pts)")
    ax.scatter(t, bds_tas_raw_kt, color="#e74c3c", s=8, alpha=0.5, label=f"BDS TAS raw ({n_raw} pts)")
    ax.scatter(t, bds_tas_clean_kt, color="#27ae60", s=10, alpha=0.7, marker="x",
               label=f"BDS TAS clean ({n_clean} pts)")
    ax.set_title(f"TAS sources — {fid}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("TAS (kt)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4, help="Number of flights to plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    delta_path = Path(cfg["paths"]["data_dir"]) / "flights.delta"
    dt = DeltaTable(str(delta_path))
    df_all = pl.from_arrow(dt.to_pyarrow_table())

    # Pick flights with at least 100 non-null bds_tas_kt points
    coverage = (
        df_all.group_by("meta_flight_id")
        .agg(
            pl.col("bds_tas_kt").is_not_null().sum().alias("n_bds_tas"),
            pl.len().alias("n_total"),
        )
        .filter(pl.col("n_bds_tas") > 100)
        .sort("n_bds_tas", descending=True)
    )
    print(f"Flights with >100 bds_tas points: {len(coverage)}")

    if len(coverage) == 0:
        print("No flight with sufficient bds_tas coverage — falling back to top-coverage flights")
        coverage = (
            df_all.group_by("meta_flight_id")
            .agg(pl.col("bds_tas_kt").is_not_null().sum().alias("n_bds_tas"))
            .sort("n_bds_tas", descending=True)
        )

    rng = np.random.default_rng(args.seed)
    candidates = coverage.head(min(50, len(coverage)))["meta_flight_id"].to_numpy()
    sampled = rng.choice(candidates, size=min(args.n, len(candidates)), replace=False)

    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fid in sampled:
        df_flight = df_all.filter(pl.col("meta_flight_id") == fid)
        n_raw = df_flight["bds_tas_kt"].is_not_null().sum() if "bds_tas_kt" in df_flight.columns else 0
        n_clean = (
            df_flight["bds_tas_kt_clean"].is_not_null().sum()
            if "bds_tas_kt_clean" in df_flight.columns
            else 0
        )
        print(f"\n{fid}: {len(df_flight)} rows | bds_tas raw={n_raw} clean={n_clean}")
        plot_flight(df_flight, out_dir / f"tas_clean_{fid}.png")


if __name__ == "__main__":
    main()
