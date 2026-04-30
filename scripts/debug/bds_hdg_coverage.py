"""Per-flight coverage of bds_hdg_deg on the full Delta dataset.

Étape 1.2 -- mesure si le fallback `track - drift` est critique ou marginal.

Usage:
    uv run python scripts/debug/bds_hdg_coverage.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable

JBU_FLIGHT = "a88774_JBU493_s0"


def compute_per_flight(df: pl.DataFrame) -> pl.DataFrame:
    """One row per flight: n_total, n_bds_valid, coverage, aircraft_type."""
    valid = pl.col("bds_hdg_deg").is_not_null() & pl.col("bds_hdg_deg").is_finite()
    return (
        df.group_by("meta_flight_id")
        .agg(
            pl.len().alias("n_total"),
            valid.sum().alias("n_bds_valid"),
            pl.col("meta_aircraft_type").first().alias("aircraft_type"),
        )
        .with_columns(
            (pl.col("n_bds_valid") / pl.col("n_total")).alias("coverage"),
        )
        .sort("meta_flight_id")
    )


def per_type_stats(per_flight: pl.DataFrame) -> list[dict]:
    """Per aircraft type: n_flights, mean/median cov, frac thresholds."""
    rows = []
    for ac_type, sub in per_flight.group_by("aircraft_type"):
        cov = sub["coverage"].to_numpy()
        rows.append(
            {
                "aircraft_type": ac_type[0] if isinstance(ac_type, tuple) else ac_type,
                "n_flights": int(sub.height),
                "mean_cov": float(np.mean(cov)),
                "median_cov": float(np.median(cov)),
                "frac_zero": float(np.mean(cov == 0.0)),
                "frac_above_0.5": float(np.mean(cov > 0.5)),
                "frac_above_0.9": float(np.mean(cov > 0.9)),
            }
        )
    rows.sort(key=lambda r: r["n_flights"], reverse=True)
    return rows


def plot_histogram(coverage: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(coverage, bins=np.linspace(0, 1, 51), color="tab:blue", edgecolor="black", alpha=0.8)
    for thr, color, label in [
        (0.0, "tab:red", "0%"),
        (0.5, "tab:orange", "50%"),
        (0.9, "tab:green", "90%"),
    ]:
        ax.axvline(thr, color=color, ls="--", lw=1.5, label=f"thr {label}")
    ax.set_xlabel("Per-flight bds_hdg coverage")
    ax.set_ylabel("Number of flights")
    ax.set_title(f"BDS heading coverage per flight (n={coverage.size})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_by_type(per_flight: pl.DataFrame, out: Path) -> None:
    counts = (
        per_flight.group_by("aircraft_type")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    top = counts.head(15)["aircraft_type"].to_list()
    data = [per_flight.filter(pl.col("aircraft_type") == t)["coverage"].to_numpy() for t in top]
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=top, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:cyan")
        patch.set_alpha(0.6)
    ax.set_ylabel("Per-flight bds_hdg coverage")
    ax.set_xlabel("Aircraft type (top 15 by flight count)")
    ax.set_title("BDS heading coverage by aircraft type")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_vs_length(per_flight: pl.DataFrame, out: Path) -> None:
    n_total = per_flight["n_total"].to_numpy()
    cov = per_flight["coverage"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(n_total, cov, s=8, alpha=0.4, color="tab:purple")
    ax.set_xlabel("Flight length (rows)")
    ax.set_ylabel("BDS heading coverage")
    ax.set_title("Coverage vs flight length")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def verdict(frac_zero: float, frac_below_0_5: float, frac_above_0_9: float) -> str:
    if frac_zero > 0.10 or frac_below_0_5 > 0.30:
        return "fallback critical"
    if frac_above_0_9 > 0.80:
        return "fallback marginal"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=Path("data/flights.delta"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/figures/lateral_step1/bds_coverage"),
    )
    args = parser.parse_args()

    if not args.delta.exists():
        print(f"Delta table not found: {args.delta}", file=sys.stderr)
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dt = DeltaTable(str(args.delta))
    df = pl.DataFrame(
        dt.to_pyarrow_table(
            columns=["meta_flight_id", "bds_hdg_deg", "meta_aircraft_type"]
        )
    )
    print(f"Loaded {df.height} rows / {df['meta_flight_id'].n_unique()} flights")

    per_flight = compute_per_flight(df)
    cov = per_flight["coverage"].to_numpy()
    n_total = per_flight["n_total"].to_numpy()
    n_valid = per_flight["n_bds_valid"].to_numpy()

    frac_zero = float(np.mean(cov == 0.0))
    frac_above_0_5 = float(np.mean(cov > 0.5))
    frac_above_0_9 = float(np.mean(cov > 0.9))
    frac_below_0_5 = float(np.mean(cov < 0.5))
    median_cov = float(np.median(cov))
    mean_cov = float(np.mean(cov))
    sample_weighted = float(n_valid.sum() / n_total.sum())

    by_type = per_type_stats(per_flight)

    v = verdict(frac_zero, frac_below_0_5, frac_above_0_9)

    stats = {
        "n_flights": int(per_flight.height),
        "n_rows_total": int(n_total.sum()),
        "n_rows_bds_valid": int(n_valid.sum()),
        "sample_weighted_coverage": sample_weighted,
        "flight_weighted_mean_coverage": mean_cov,
        "flight_weighted_median_coverage": median_cov,
        "frac_zero": frac_zero,
        "frac_below_0.5": frac_below_0_5,
        "frac_above_0.5": frac_above_0_5,
        "frac_above_0.9": frac_above_0_9,
        "verdict": v,
        "by_aircraft_type": by_type,
    }

    plot_histogram(cov, args.out_dir / "coverage_histogram.png")
    plot_by_type(per_flight, args.out_dir / "coverage_by_aircraft_type.png")
    plot_vs_length(per_flight, args.out_dir / "coverage_vs_length.png")

    with open(args.out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    zero_flights = (
        per_flight.filter(pl.col("coverage") == 0.0)["meta_flight_id"].to_list()
    )
    full_flights = (
        per_flight.filter(pl.col("coverage") > 0.99)["meta_flight_id"].to_list()
    )
    (args.out_dir / "flights_zero_coverage.txt").write_text(
        "\n".join(sorted(zero_flights)) + ("\n" if zero_flights else "")
    )
    (args.out_dir / "flights_full_coverage.txt").write_text(
        "\n".join(sorted(full_flights)) + ("\n" if full_flights else "")
    )

    # Console summary
    print()
    print("=== BDS heading coverage summary ===")
    print(f"Flights              : {per_flight.height}")
    print(f"Rows total           : {n_total.sum():,}")
    print(f"Rows BDS valid       : {n_valid.sum():,}")
    print(f"Sample-weighted cov  : {sample_weighted:.4f}")
    print(f"Flight-weighted mean : {mean_cov:.4f}")
    print(f"Flight-weighted med  : {median_cov:.4f}")
    print(f"frac_zero            : {frac_zero:.4f}")
    print(f"frac_below_0.5       : {frac_below_0_5:.4f}")
    print(f"frac_above_0.5       : {frac_above_0_5:.4f}")
    print(f"frac_above_0.9       : {frac_above_0_9:.4f}")
    print()
    print("=== Top 10 aircraft types by flight count ===")
    print(f"{'type':<10} {'n':>5} {'mean':>6} {'median':>7} {'fz':>6} {'>0.5':>6} {'>0.9':>6}")
    for r in by_type[:10]:
        print(
            f"{str(r['aircraft_type']):<10} {r['n_flights']:>5d} "
            f"{r['mean_cov']:>6.3f} {r['median_cov']:>7.3f} "
            f"{r['frac_zero']:>6.3f} {r['frac_above_0.5']:>6.3f} {r['frac_above_0.9']:>6.3f}"
        )
    print()
    jbu_in_zero = JBU_FLIGHT in zero_flights
    print(f"JBU sanity ({JBU_FLIGHT}) coverage == 0 : {jbu_in_zero}")
    if not jbu_in_zero:
        row = per_flight.filter(pl.col("meta_flight_id") == JBU_FLIGHT)
        if row.height:
            print(f"  actual coverage: {float(row['coverage'][0]):.4f}")
    print()
    print(f"VERDICT: {v}")
    print(f"Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
