"""Plot state vs target columns for the adsb architecture.

For each flight, draws three panels stacked vertically:

1. Altitude:    raw_alt_ft           vs fdm_alt_target_ft
2. TAS:         bds_tas_from_cas_kt  vs fdm_tas_target_kt
3. Gamma:       fdm_gamma_rad        vs fdm_gamma_target_rad

State is plotted in blue, target in red. Output: one PNG per flight under
``data/figures/state_targets/{flight_id}.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from deltalake import DeltaTable

DEFAULT_FLIGHTS = [
    "738284_ISR826_s0",
    "40666a_EZY78QT_s0",
    "4a0443_BTI7FR_s0",
    "a88774_JBU493_s0",
    "44006e_AUA445_s0",
]

PANELS: list[tuple[str, str, str, str]] = [
    ("Altitude [ft]", "raw_alt_ft", "fdm_alt_target_ft", "altitude"),
    ("TAS [kt]", "bds_tas_from_cas_kt", "fdm_tas_target_kt", "tas"),
    ("Gamma [rad]", "fdm_gamma_rad", "fdm_gamma_target_rad", "gamma"),
]


def plot_flight(df: pl.DataFrame, flight_id: str, out_dir: Path) -> Path | None:
    flight = df.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
    if flight.height == 0:
        print(f"skip {flight_id}: empty", file=sys.stderr)
        return None

    ts = flight["raw_timestamp"].to_list()
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(14, 9), sharex=True)

    meta = flight.row(0, named=True)
    fig.suptitle(
        f"{flight_id}  -  {meta.get('meta_departure', '?')} -> {meta.get('meta_arrival', '?')}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (ylabel, state_col, target_col, _) in zip(axes, PANELS, strict=False):
        if state_col in flight.columns:
            ax.plot(
                ts,
                flight[state_col].to_list(),
                color="tab:blue",
                lw=1.0,
                label=f"state ({state_col})",
            )
        if target_col in flight.columns:
            ax.plot(
                ts,
                flight[target_col].to_list(),
                color="tab:red",
                lw=1.5,
                label=f"target ({target_col})",
            )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (UTC)")
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    out_path = out_dir / f"{flight_id}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flight",
        action="append",
        help="Flight ID to plot (repeatable). Defaults to 5 reference flights.",
    )
    parser.add_argument(
        "--delta",
        type=Path,
        default=Path("data/flights.delta"),
        help="Path to the Delta table (default: data/flights.delta).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/figures/state_targets"),
        help="Output directory (default: data/figures/state_targets).",
    )
    args = parser.parse_args()

    if not args.delta.exists():
        print(f"Delta table not found: {args.delta}", file=sys.stderr)
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dt = DeltaTable(str(args.delta))
    df = pl.DataFrame(dt.to_pyarrow_table())

    flights = args.flight or DEFAULT_FLIGHTS
    for fid in flights:
        plot_flight(df, fid, args.out_dir)


if __name__ == "__main__":
    main()
