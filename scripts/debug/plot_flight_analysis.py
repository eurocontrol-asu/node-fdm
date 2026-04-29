"""Flight analysis — multi-panel grouped chart for a single flight."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import polars as pl

# ---------------------------------------------------------------------------
# Column groups: (group_label, [(col, display_name), ...])
# ---------------------------------------------------------------------------
GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Altitude (ft)",
        [
            ("raw_alt_ft", "Altitude"),
            ("fdm_alt_sel_ft", "Alt Selected (segments)"),
            ("fdm_alt_target_ft", "Alt Target (bfill)"),
        ],
    ),
    (
        "Speed — Mach",
        [
            ("era_mach", "Mach"),
            ("fdm_mach_sel", "Mach Selected (segments)"),
        ],
    ),
    (
        "Speed — CAS (kt)",
        [
            ("bds_ias_kt", "CAS (IAS)"),
            ("fdm_cas_sel_kt", "CAS Selected (segments)"),
            ("fdm_cas_target_kt", "CAS Target (bfill)"),
        ],
    ),
]

WIDTH = 900
HEIGHT = 150


def _build_chart(df: pl.DataFrame, flight_id: str) -> alt.VConcatChart:
    """Build a vertically concatenated chart with one panel per group."""
    # Add a row index as x-axis (seconds from start)
    df = df.with_row_index("_idx")
    panels = []

    for group_label, cols in GROUPS:
        available = [(c, label) for c, label in cols if c in df.columns]
        if not available:
            continue

        # Melt selected columns into long format
        col_names = [c for c, _ in available]
        label_map = dict(available)

        melted = (
            df.select(["_idx", *col_names])
            .unpivot(index="_idx", on=col_names, variable_name="series", value_name="value")
            .with_columns(pl.col("series").replace(label_map))
            .to_pandas()
        )

        panel = (
            alt.Chart(melted)
            .mark_line(strokeWidth=1.2, opacity=0.85)
            .encode(
                x=alt.X("_idx:Q", title="timestep"),
                y=alt.Y("value:Q", title=group_label),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="right")),
                tooltip=["_idx:Q", "series:N", "value:Q"],
            )
            .properties(width=WIDTH, height=HEIGHT, title=group_label)
        )
        panels.append(panel)

    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="independent")
        .properties(
            title=f"Flight Analysis — {flight_id}",
        )
    )
    return chart  # type: ignore[no-any-return]


def main() -> None:
    delta_path = Path("data/flights.delta")
    if not delta_path.exists():
        print("Delta table not found at data/flights.delta", file=sys.stderr)
        raise SystemExit(1)

    full_df = pl.read_delta(str(delta_path))

    if len(sys.argv) < 2:
        # Pick a random flight
        import random

        flight_ids = full_df["meta_flight_id"].unique().to_list()
        flight_id = random.choice(flight_ids)  # noqa: S311
    else:
        flight_id = sys.argv[1]

    df = full_df.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
    if len(df) == 0:
        print(f"Flight not found: {flight_id}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Flight: {flight_id} — {len(df)} rows, {len(df.columns)} columns")

    chart = _build_chart(df, flight_id)
    out = Path("data/figures") / f"flight_analysis_{flight_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(out))
    print(f"✅ Saved to {out}")


if __name__ == "__main__":
    main()
