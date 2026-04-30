"""Quantify the cruise / climb / descent / transition composition of the
A320 ADS-B training set used by ``node_adsb_v1_A320``.

Goal: inform the F1 (per-head input routing) decision documented in
``data/mardown/per_head_input_routing.md``. F1 cuts ``tas_diff -> n_z``
and ``gamma_diff -> a_spec``, which is structurally invalid in
SPEED_ON_PITCH phases (climb under THR REF, descent idle) where pitch
drives speed. Before committing to F1, we need the fraction of the
training set spent in those phases.

Inputs
------
- Delta table at ``data/flights.delta`` (read-only).
- Filter: ``fdm_flag_valid AND meta_aircraft_type == 'A320' AND
  meta_split == 'train'`` (matches ``node_fdm_pipeline.commands.train``).

Method
------
Per-row classification:

- **cruise**     : |gamma| < 0.5 deg AND |alt_rate| < 200 ft/min
- **climb**      : gamma > +1 deg OR alt_rate > +500 ft/min
- **descent**    : gamma < -1 deg OR alt_rate < -500 ft/min
- **transition** : everything else (level-off, shallow, accel/decel)

Sub-classification of climb/descent:

- **climb_saturated**   : alt_rate > 1500 ft/min  -> likely SPEED_ON_PITCH
- **descent_saturated** : alt_rate < -1000 ft/min -> likely SPEED_ON_PITCH idle
- **shallow**           : everything else inside climb/descent

Outputs a markdown report on stdout (or to ``--output``).

Usage
-----
    uv run python scripts/debug/dataset_regime_stats.py
    uv run python scripts/debug/dataset_regime_stats.py --output report.md
    uv run python scripts/debug/dataset_regime_stats.py --typecode A320 --split train
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import polars as pl

__all__ = ["main", "compute_stats"]

# Thresholds (documented in report)
GAMMA_CRUISE_RAD = math.radians(0.5)   # 0.5 deg
GAMMA_CLIMB_RAD = math.radians(1.0)    # +1 deg
GAMMA_DESCENT_RAD = math.radians(-1.0)  # -1 deg

# Vertical-rate thresholds (m/s) — convert from ft/min: 1 ft/min = 0.00508 m/s
FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS         # 1.016 m/s
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS          # 2.54 m/s
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS       # -2.54 m/s
ALT_RATE_CLIMB_SAT = 1500 * FTMIN_TO_MS     # 7.62 m/s
ALT_RATE_DESCENT_SAT = -1000 * FTMIN_TO_MS  # -5.08 m/s


def _classify_expr() -> pl.Expr:
    """Build a polars expression that maps each row to a regime label."""
    gamma = pl.col("fdm_gamma_rad")
    alt_rate = pl.col("fdm_d_alt_ms")

    is_cruise = (gamma.abs() < GAMMA_CRUISE_RAD) & (alt_rate.abs() < ALT_RATE_CRUISE)
    is_climb = (gamma > GAMMA_CLIMB_RAD) | (alt_rate > ALT_RATE_CLIMB)
    is_descent = (gamma < GAMMA_DESCENT_RAD) | (alt_rate < ALT_RATE_DESCENT)

    # Order matters: cruise wins (strict band), then climb/descent, else transition.
    return (
        pl.when(is_cruise)
        .then(pl.lit("cruise"))
        .when(is_climb)
        .then(pl.lit("climb"))
        .when(is_descent)
        .then(pl.lit("descent"))
        .otherwise(pl.lit("transition"))
        .alias("regime")
    )


def _saturation_expr() -> pl.Expr:
    """Sub-label for climb/descent rows."""
    alt_rate = pl.col("fdm_d_alt_ms")
    return (
        pl.when(alt_rate > ALT_RATE_CLIMB_SAT)
        .then(pl.lit("climb_saturated"))
        .when(alt_rate < ALT_RATE_DESCENT_SAT)
        .then(pl.lit("descent_saturated"))
        .otherwise(pl.lit("shallow"))
        .alias("saturation")
    )


def compute_stats(
    delta_path: Path,
    *,
    typecode: str = "A320",
    split: str = "train",
) -> dict:
    """Scan the Delta table and compute regime statistics.

    Returns a dict ready to be rendered as markdown.
    """
    needed = [
        "meta_aircraft_type",
        "meta_split",
        "meta_flight_id",
        "fdm_flag_valid",
        "fdm_gamma_rad",
        "fdm_d_alt_ms",
        "fdm_tas_diff_ms",
        "fdm_gamma_diff_rad",
    ]

    lf = (
        pl.scan_delta(str(delta_path))
        .filter(pl.col("fdm_flag_valid"))
        .filter(pl.col("meta_aircraft_type") == typecode)
        .filter(pl.col("meta_split") == split)
        .select(needed)
        .with_columns([_classify_expr(), _saturation_expr()])
    )

    # Drop rows where gamma_rad / d_alt_ms is null (cannot classify).
    lf = lf.drop_nulls(["fdm_gamma_rad", "fdm_d_alt_ms"])
    # Polars drop_nulls does not drop NaN floats. Filter them out explicitly:
    lf = lf.filter(
        ~pl.col("fdm_gamma_rad").is_nan() & ~pl.col("fdm_d_alt_ms").is_nan()
    )

    df = lf.collect()
    n_total = df.height
    if n_total == 0:
        return {"error": "no rows after filtering", "delta_path": str(delta_path)}

    # Per-regime sample counts.
    regime_counts = (
        df.group_by("regime").agg(pl.len().alias("n")).sort("regime")
    )

    # Per-regime |tas_diff| and |gamma_diff| stats.
    # Use a filtered frame where NaN diffs are excluded, so quantiles are
    # not poisoned. Aggregate using filtered expressions per column so
    # tas/gamma can be NaN in different rows.
    tas_ok = ~pl.col("fdm_tas_diff_ms").is_nan()
    gam_ok = ~pl.col("fdm_gamma_diff_rad").is_nan()
    per_regime_diffs = df.group_by("regime").agg(
        [
            pl.col("fdm_tas_diff_ms").filter(tas_ok).abs().mean().alias("tas_diff_abs_mean"),
            pl.col("fdm_tas_diff_ms").filter(tas_ok).abs().median().alias("tas_diff_abs_median"),
            pl.col("fdm_tas_diff_ms").filter(tas_ok).abs().quantile(0.95).alias("tas_diff_abs_p95"),
            pl.col("fdm_tas_diff_ms").filter(tas_ok).abs().quantile(0.99).alias("tas_diff_abs_p99"),
            pl.col("fdm_tas_diff_ms").filter(tas_ok).count().alias("n_tas_known"),
            pl.col("fdm_gamma_diff_rad").filter(gam_ok).abs().mean().alias("gamma_diff_abs_mean"),
            pl.col("fdm_gamma_diff_rad").filter(gam_ok).abs().median().alias("gamma_diff_abs_median"),
            pl.col("fdm_gamma_diff_rad").filter(gam_ok).abs().quantile(0.95).alias("gamma_diff_abs_p95"),
            pl.col("fdm_gamma_diff_rad").filter(gam_ok).abs().quantile(0.99).alias("gamma_diff_abs_p99"),
            pl.col("fdm_gamma_diff_rad").filter(gam_ok).count().alias("n_gamma_known"),
            pl.len().alias("n"),
        ],
    ).sort("regime")

    # Saturation breakdown of climb/descent.
    cd_df = df.filter(pl.col("regime").is_in(["climb", "descent"]))
    sat_counts = (
        cd_df.group_by(["regime", "saturation"])
        .agg(pl.len().alias("n"))
        .sort(["regime", "saturation"])
    )

    # tas_diff / gamma_diff distribution in saturated climb (the
    # "would-F1-cut-real-signal" question).
    sat_climb = df.filter(
        (pl.col("regime") == "climb") & (pl.col("saturation") == "climb_saturated")
    )
    sat_descent = df.filter(
        (pl.col("regime") == "descent") & (pl.col("saturation") == "descent_saturated")
    )

    def _joint_stats(frame: pl.DataFrame) -> dict:
        if frame.height == 0:
            return {"n": 0}
        tas = frame.filter(~pl.col("fdm_tas_diff_ms").is_nan())["fdm_tas_diff_ms"]
        gam = frame.filter(~pl.col("fdm_gamma_diff_rad").is_nan())["fdm_gamma_diff_rad"]
        out: dict = {"n": frame.height, "n_tas_known": tas.len(), "n_gamma_known": gam.len()}
        if tas.len() > 0:
            out["tas_diff_abs_mean"] = float(tas.abs().mean())
            out["tas_diff_abs_p95"] = float(tas.abs().quantile(0.95))
            out["tas_diff_abs_p99"] = float(tas.abs().quantile(0.99))
            out["frac_tas_diff_gt_1ms"] = float((tas.abs() > 1.0).mean())
        if gam.len() > 0:
            out["gamma_diff_abs_mean"] = float(gam.abs().mean())
            out["gamma_diff_abs_p95"] = float(gam.abs().quantile(0.95))
            out["gamma_diff_abs_p99"] = float(gam.abs().quantile(0.99))
            out["frac_gamma_diff_gt_0p5deg"] = float(
                (gam.abs() > math.radians(0.5)).mean()
            )
        return out

    # Per-trajectory dominant regime.
    flight_regime = (
        df.group_by(["meta_flight_id", "regime"])
        .agg(pl.len().alias("n"))
        .sort(["meta_flight_id", "n"], descending=[False, True])
    )
    # Take first row per flight_id (highest n) -> dominant regime.
    flight_dominant = flight_regime.group_by("meta_flight_id").agg(
        pl.col("regime").first().alias("dominant_regime"),
    )
    flight_dom_counts = (
        flight_dominant.group_by("dominant_regime")
        .agg(pl.len().alias("n_flights"))
        .sort("dominant_regime")
    )
    n_flights = flight_dominant.height

    return {
        "delta_path": str(delta_path),
        "typecode": typecode,
        "split": split,
        "n_total": n_total,
        "n_flights": n_flights,
        "regime_counts": regime_counts.to_dicts(),
        "per_regime_diffs": per_regime_diffs.to_dicts(),
        "sat_counts": sat_counts.to_dicts(),
        "flight_dom_counts": flight_dom_counts.to_dicts(),
        "sat_climb_stats": _joint_stats(sat_climb),
        "sat_descent_stats": _joint_stats(sat_descent),
    }


def _fmt_pct(num: int, denom: int) -> str:
    return f"{100.0 * num / denom:.2f}%" if denom else "n/a"


def render_markdown(stats: dict) -> str:  # noqa: C901 - report formatter, linear
    if "error" in stats:
        return f"# Dataset regime stats — ERROR\n\n{stats}\n"

    n_total = stats["n_total"]
    n_flights = stats["n_flights"]

    lines: list[str] = []
    lines.append("# A320 ADS-B training-set regime composition")
    lines.append("")
    lines.append(f"- Delta table : `{stats['delta_path']}`")
    lines.append(f"- Filter      : `fdm_flag_valid` AND `meta_aircraft_type == '{stats['typecode']}'` AND `meta_split == '{stats['split']}'`")
    lines.append(f"- Samples     : **{n_total:,}** rows")
    lines.append(f"- Trajectories: **{n_flights:,}** distinct `meta_flight_id`")
    lines.append("")
    lines.append("## Methodology — thresholds")
    lines.append("")
    lines.append("Per-row classification (gamma in radians from `fdm_gamma_rad`, alt rate in m/s from `fdm_d_alt_ms`).")
    lines.append("")
    lines.append("| Class      | Rule |")
    lines.append("|---|---|")
    lines.append(f"| cruise     | `\\|gamma\\| < {math.degrees(GAMMA_CRUISE_RAD):.2f} deg` AND `\\|alt_rate\\| < 200 ft/min` |")
    lines.append("| climb      | `gamma > +1 deg` OR `alt_rate > +500 ft/min` |")
    lines.append("| descent    | `gamma < -1 deg` OR `alt_rate < -500 ft/min` |")
    lines.append("| transition | otherwise (level-off, shallow climb/descent, accel/decel cruise) |")
    lines.append("")
    lines.append("Saturation refines climb/descent only:")
    lines.append("")
    lines.append("| Sub-class           | Rule                              | Likely AFCS mode      |")
    lines.append("|---|---|---|")
    lines.append("| climb_saturated     | `alt_rate > 1500 ft/min`          | SPEED_ON_PITCH (THR REF) |")
    lines.append("| descent_saturated   | `alt_rate < -1000 ft/min`         | SPEED_ON_PITCH (IDLE)    |")
    lines.append("| shallow             | otherwise within climb/descent    | likely SPEED_ON_THROTTLE |")
    lines.append("")
    lines.append("Order matters: cruise wins on its strict band first, then climb/descent, else transition.")
    lines.append("Rows with null `fdm_gamma_rad` or `fdm_d_alt_ms` are dropped before classification.")
    lines.append("")

    # Distribution table.
    lines.append("## Distribution of regimes (samples)")
    lines.append("")
    lines.append("| Regime     | Samples       | % of total |")
    lines.append("|---|---:|---:|")
    counts_by_regime = {row["regime"]: row["n"] for row in stats["regime_counts"]}
    for regime in ["cruise", "climb", "descent", "transition"]:
        n = counts_by_regime.get(regime, 0)
        lines.append(f"| {regime:<10} | {n:>13,} | {_fmt_pct(n, n_total):>10} |")
    lines.append(f"| **total**  | {n_total:>13,} | {'100.00%':>10} |")
    lines.append("")

    # Trajectory dominance.
    lines.append("## Distribution of trajectories (dominant regime per flight)")
    lines.append("")
    lines.append("| Dominant regime | Flights | % of fleet |")
    lines.append("|---|---:|---:|")
    fdom = {row["dominant_regime"]: row["n_flights"] for row in stats["flight_dom_counts"]}
    for regime in ["cruise", "climb", "descent", "transition"]:
        n = fdom.get(regime, 0)
        lines.append(f"| {regime:<15} | {n:>7,} | {_fmt_pct(n, n_flights):>10} |")
    lines.append("")

    # Per-regime diff stats.
    lines.append("## |tas_diff| and |gamma_diff| conditioned by regime")
    lines.append("")
    lines.append("`tas_diff = tas_target − tas` (m/s). `gamma_diff = gamma_target − gamma` (rad).")
    lines.append("These are exactly the features F1 would cut from the wrong head.")
    lines.append("")
    lines.append("Stats below exclude rows where the consigne is unknown")
    lines.append("(NaN `tas_diff` / `gamma_diff`). Counts `n_tas_known` and")
    lines.append("`n_gamma_known` report how many rows of the regime had a known target.")
    lines.append("")
    lines.append("| Regime     | n              | n_tas_known | mean\\|tas_diff\\| | p95\\|tas_diff\\| | p99\\|tas_diff\\| | n_gamma_known | mean\\|gamma_diff\\| (deg) | p95\\|gamma_diff\\| (deg) | p99\\|gamma_diff\\| (deg) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def _g(d: dict, key: str, fmt: str = ".3f", deg: bool = False) -> str:
        v = d.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        if deg:
            v = math.degrees(v)
        return format(v, fmt)

    for row in stats["per_regime_diffs"]:
        lines.append(
            f"| {row['regime']:<10} | {row['n']:>14,} | "
            f"{row['n_tas_known']:>11,} | "
            f"{_g(row, 'tas_diff_abs_mean')} | "
            f"{_g(row, 'tas_diff_abs_p95')} | "
            f"{_g(row, 'tas_diff_abs_p99')} | "
            f"{row['n_gamma_known']:>13,} | "
            f"{_g(row, 'gamma_diff_abs_mean', '.4f', deg=True)} | "
            f"{_g(row, 'gamma_diff_abs_p95', '.4f', deg=True)} | "
            f"{_g(row, 'gamma_diff_abs_p99', '.4f', deg=True)} |"
        )
    lines.append("")

    # Saturation.
    lines.append("## Climb / descent saturation breakdown")
    lines.append("")
    lines.append("| Regime  | Saturation         | Samples       | % of regime | % of total |")
    lines.append("|---|---|---:|---:|---:|")
    sat_by_regime: dict[str, dict[str, int]] = {}
    for row in stats["sat_counts"]:
        sat_by_regime.setdefault(row["regime"], {})[row["saturation"]] = row["n"]
    for regime in ["climb", "descent"]:
        regime_total = counts_by_regime.get(regime, 0)
        for sat in ["climb_saturated", "descent_saturated", "shallow"]:
            n = sat_by_regime.get(regime, {}).get(sat, 0)
            if n == 0 and sat in {"climb_saturated", "descent_saturated"} and (
                (regime == "climb" and sat != "climb_saturated")
                or (regime == "descent" and sat != "descent_saturated")
            ):
                continue
            lines.append(
                f"| {regime:<7} | {sat:<18} | {n:>13,} | "
                f"{_fmt_pct(n, regime_total):>11} | "
                f"{_fmt_pct(n, n_total):>10} |"
            )
    lines.append("")

    sat_climb_total = sat_by_regime.get("climb", {}).get("climb_saturated", 0)
    sat_descent_total = sat_by_regime.get("descent", {}).get("descent_saturated", 0)
    sat_total = sat_climb_total + sat_descent_total
    lines.append(
        f"**Saturated climb + descent = {sat_total:,} samples = "
        f"{_fmt_pct(sat_total, n_total)} of training set.**",
    )
    lines.append("")

    # Joint distribution in saturated climb/descent.
    lines.append("## (tas_diff, gamma_diff) joint stats in saturated phases")
    lines.append("")
    lines.append("Question: in saturated climb/descent (likely SPEED_ON_PITCH), is")
    lines.append("`tas_diff` and `gamma_diff` actually informative? If yes, F1")
    lines.append("(per-head input routing) cuts real signal there.")
    lines.append("")
    lines.append("Counts `n_tas_known` / `n_gamma_known` are the rows of the phase")
    lines.append("with a known target (NaN `tas_diff` / `gamma_diff` excluded).")
    lines.append("")
    lines.append("| Phase             | n           | n_tas_known | mean\\|tas_diff\\| | p95\\|tas_diff\\| | p99\\|tas_diff\\| | %\\|tas_diff\\|>1m/s | n_gamma_known | mean\\|gamma_diff\\| (deg) | p95\\|gamma_diff\\| (deg) | %\\|gamma_diff\\|>0.5deg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    def _v(d: dict, key: str, fmt: str = ".3f", deg: bool = False, pct: bool = False) -> str:
        v = d.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        if pct:
            return f"{100*v:.2f}%"
        if deg:
            v = math.degrees(v)
        return format(v, fmt)

    for label, key in [
        ("climb_saturated", "sat_climb_stats"),
        ("descent_saturated", "sat_descent_stats"),
    ]:
        s = stats[key]
        if s.get("n", 0) == 0:
            lines.append(f"| {label:<17} | {0:>11} | 0 | n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label:<17} | {s['n']:>11,} | "
            f"{s.get('n_tas_known', 0):>11,} | "
            f"{_v(s, 'tas_diff_abs_mean')} | "
            f"{_v(s, 'tas_diff_abs_p95')} | "
            f"{_v(s, 'tas_diff_abs_p99')} | "
            f"{_v(s, 'frac_tas_diff_gt_1ms', pct=True)} | "
            f"{s.get('n_gamma_known', 0):>13,} | "
            f"{_v(s, 'gamma_diff_abs_mean', '.4f', deg=True)} | "
            f"{_v(s, 'gamma_diff_abs_p95', '.4f', deg=True)} | "
            f"{_v(s, 'frac_gamma_diff_gt_0p5deg', pct=True)} |"
        )
    lines.append("")

    # Verdict.
    lines.append("## Verdict")
    lines.append("")
    pct_sat = 100.0 * sat_total / n_total
    if pct_sat < 5.0:
        verdict = (
            f"Saturated climb/descent fraction is **{pct_sat:.2f}% < 5%**. "
            "F1 (per-head input routing) is acceptable; the model can be "
            "labelled cruise-only. The structural loss in saturated phases "
            "is bounded by the small population there."
        )
    elif pct_sat > 15.0:
        verdict = (
            f"Saturated climb/descent fraction is **{pct_sat:.2f}% > 15%**. "
            "F1 would systematically degrade a substantial share of the data. "
            "Prefer a soft, cruise-conditional regularisation on "
            "`d a_spec / d gamma_diff` and `d n_z / d tas_diff` rather than "
            "an architectural cut."
        )
    else:
        verdict = (
            f"Saturated climb/descent fraction is **{pct_sat:.2f}% (between 5% and 15%)**. "
            "F1 is a measurable trade-off: real signal in ~{pct:.0f}% of rows "
            "would be cut. Recommend an A/B retrain (F1 vs. soft "
            "regularisation) and pick the winner on validation NLL conditioned "
            "by regime."
        ).format(pct=pct_sat)
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta",
        type=Path,
        default=Path("data/flights.delta"),
        help="Path to the Delta table (default: data/flights.delta).",
    )
    parser.add_argument("--typecode", default="A320")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this file instead of stdout.",
    )
    args = parser.parse_args()

    if not args.delta.exists():
        print(f"ERROR: delta path {args.delta} does not exist", file=sys.stderr)
        return 1

    stats = compute_stats(args.delta, typecode=args.typecode, split=args.split)
    report = render_markdown(stats)
    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
