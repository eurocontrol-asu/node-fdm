"""Phase 10 — AC7 Seymour sweep : does aircraft-age correction improve AC7 vs QAR FUEL_FF ?

Validation-only (R1) : QAR cruise samples used to score the analytical
mdot_f chain with and without the Seymour age-deterioration correction.
Re-uses the kinematic pipeline of
:mod:`phase4_fuelflow_qar_validation` (cruise filter → T-observable →
η_PS → Eq 19) but replays the η_PS chain with a per-sample
``(1 − α · ACFT_AGE)`` multiplicative correction.

The Seymour α coefficient sweep tests :
    α = 0.000   → v17 baseline (no age correction).
    α = 0.0035  → v20 default (literature mid-fleet).
    α = 0.001 / 0.002 / 0.005 / 0.007 / 0.010 → bracket.

Best α maximises the Pearson correlation `corr(mdot_f_pred, FUEL__FF_QAR)`
on the 526 A320 cruise-stable cohort. Reports :

  * AC7 corr per α.
  * bias `median(mdot_f_pred / mdot_f_qar)`.
  * median / p99 absolute relative error.
  * Δ corr vs baseline (α = 0).

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    unset VIRTUAL_ENV
    uv run python scripts/_validation/phase10_seymour_ac7_sweep.py \\
        --report data/models/_comparison/phase10_seymour_ac7_sweep.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Re-use the existing QAR validation pipeline (cruise filter + analytical
# η_PS + Eq 19) by importing from the sibling script in the parent
# scripts/ directory.
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from phase4_fuelflow_qar_validation import (  # noqa: E402
    LCV_KEROSENE,
    QAR_DIR,
    eta_o_np,
    process_flight,
)


def process_flight_with_age(parquet_path: Path) -> dict[str, np.ndarray] | None:
    """Wrap :func:`process_flight` to also capture the per-sample ACFT_AGE.

    The QAR parquet schema carries ``ACFT_AGE`` (years, computed per-flight
    from aircraft build date vs flight date). We pass it through alongside
    the kinematic arrays so the AC7 sweep can apply the Seymour
    ``(1 − α · age)`` correction per sample.
    """
    base = process_flight(parquet_path)
    if base is None:
        return None
    df = pl.read_parquet(parquet_path)
    if "ACFT_AGE" not in df.columns:
        return None
    n_target = len(base["alt_m"])
    # process_flight slices to df_cruise and aligns with mass_kg. To stay
    # consistent we re-derive the cruise mask the same way before reading
    # ACFT_AGE. Easier alternative : the ACFT_AGE column is essentially
    # constant per flight (one aircraft = one age at flight date), so we
    # just take the median over the whole parquet and broadcast.
    age_series = df["ACFT_AGE"].drop_nulls()
    if age_series.is_empty():
        return None
    age_years = float(age_series.median())
    base["age_years"] = np.full(n_target, age_years, dtype=np.float32)
    base["age_per_flight"] = np.array([age_years], dtype=np.float32)
    return base


def compute_ac7_with_seymour(
    cached: dict[str, np.ndarray],
    alpha: float,
    mode: str = "linear",
) -> dict[str, float]:
    """AC7 metrics under a Seymour age correction.

    Two modes are supported :

    - ``mode="linear"`` (v20 baseline approximation) — multiply η by
      ``(1 − α · age_years)``. α is in 1/year units. Extrapolates
      aggressively at high ages.

    - ``mode="log"`` (Seymour 2020 canonical formula) — multiply η by
      ``(1 − k · ln(age_years + 1))``. k is dimensionless ; the canonical
      Seymour coefficient is 0.0128 (i.e. fuel-flow penalty
      ``= 100 / (100 − 1.28 · ln(age + 1))``, with the 1.28 in percent
      units). k grows the penalty sublinearly with age, matching the
      industry "engine wear plateau after 10-15 yr" intuition.

    α = 0 reproduces the v14 / v17 / v19 baseline (no correction).
    """
    c_t_inst = cached["c_t_inst"]
    mach = cached["mach"]
    t_obs = cached["t_obs_n"]
    tas_ms = cached["tas_ms"]
    mdot_qar = cached["mdot_f_qar_kg_s"]
    valid_kin = cached["valid_kin"]
    age_years = cached["age_years"]

    eta_ps = eta_o_np(c_t_inst, mach)
    if mode == "linear":
        eta_corrected = eta_ps * (1.0 - alpha * age_years)
    elif mode == "log":
        eta_corrected = eta_ps * (1.0 - alpha * np.log(age_years + 1.0))
    else:
        msg = f"unknown mode {mode!r} ; expected 'linear' or 'log'."
        raise ValueError(msg)
    mdot_pred = t_obs * tas_ms / np.maximum(eta_corrected * LCV_KEROSENE, 1e-6)
    valid = valid_kin & (eta_corrected > 0.05) & (eta_corrected < 0.55)

    if int(valid.sum()) < 100:
        return {
            "alpha": alpha,
            "n": 0,
            "corr": float("nan"),
            "bias_median": float("nan"),
            "median_abs_rel_err_pct": float("nan"),
            "p99_abs_rel_err_pct": float("nan"),
        }

    mp = mdot_pred[valid]
    mq = mdot_qar[valid]
    corr = float(np.corrcoef(mp, mq)[0, 1])
    abs_rel = np.abs((mp - mq) / mq)
    return {
        "alpha": alpha,
        "n": int(valid.sum()),
        "corr": corr,
        "bias_median": float(np.median(mp / mq)),
        "median_abs_rel_err_pct": float(np.median(abs_rel)) * 100.0,
        "p99_abs_rel_err_pct": float(np.percentile(abs_rel, 99)) * 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10 — AC7 Seymour α sweep on QAR cruise cohort"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.000,0.001,0.002,0.0035,0.005,0.007,0.010",
        help="Comma-separated α values for the linear mode sweep.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="0.000,0.005,0.0128,0.020,0.0243,0.030,0.040,0.050",
        help=(
            "Comma-separated k values for the log mode sweep. "
            "Canonical Seymour 2020 is 0.0128 ; the user-cited paper "
            "value is 0.0243."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("linear", "log", "both"),
        default="both",
        help="Which sweep mode(s) to run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/models/_comparison/phase10_seymour_ac7_sweep.md"),
    )
    args = parser.parse_args()

    alphas = [float(v.strip()) for v in args.alphas.split(",")]
    ks = [float(v.strip()) for v in args.ks.split(",")]
    print(f"Mode : {args.mode}")
    if args.mode in ("linear", "both"):
        print(f"  linear α values : {alphas}")
    if args.mode in ("log", "both"):
        print(f"  log    k values : {ks}")

    paths = sorted(QAR_DIR.glob("*A320*.parquet"))
    print(f"Found {len(paths)} A320 QAR files")
    flights: list[dict[str, np.ndarray]] = []
    for i, p in enumerate(paths):
        if i % 50 == 0:
            print(f"  [{i + 1}/{len(paths)}] {p.name}")
        d = process_flight_with_age(p)
        if d is not None:
            flights.append(d)
    print(f"Used {len(flights)} flights with ACFT_AGE present.")
    if not flights:
        print("No valid flights — abort.")
        return 1

    # Cohort age stats.
    ages_per_flight = np.concatenate([f["age_per_flight"] for f in flights])
    print(
        f"\nCohort age (per flight) : "
        f"n={len(ages_per_flight)}, "
        f"min={ages_per_flight.min():.1f}, "
        f"median={np.median(ages_per_flight):.1f}, "
        f"mean={ages_per_flight.mean():.2f}, "
        f"std={ages_per_flight.std():.2f}, "
        f"max={ages_per_flight.max():.1f}"
    )

    cached = {
        "alt_m": np.concatenate([d["alt_m"] for d in flights]),
        "tas_ms": np.concatenate([d["tas_ms"] for d in flights]),
        "mach": np.concatenate([d["mach"] for d in flights]),
        "mass_kg": np.concatenate([d["mass_kg"] for d in flights]),
        "q_pa": np.concatenate([d["q_pa"] for d in flights]),
        "drag_n": np.concatenate([d["drag_n"] for d in flights]),
        "t_obs_n": np.concatenate([d["t_obs_n"] for d in flights]),
        "c_t_inst": np.concatenate([d["c_t_inst"] for d in flights]),
        "mdot_f_qar_kg_s": np.concatenate([d["mdot_f_qar_kg_s"] for d in flights]),
        "valid_kin": np.concatenate([d["valid_kin"] for d in flights]),
        "age_years": np.concatenate([d["age_years"] for d in flights]),
    }
    print(f"\nTotal cruise-stable samples : {len(cached['valid_kin'])}")

    def _sweep_one(
        values: list[float], mode: str
    ) -> list[tuple[str, dict[str, float]]]:
        label = "α" if mode == "linear" else "k"
        print(
            f"\n=== {mode.upper()} sweep ==="
            f"\n{label:>8} {'n':>9} {'AC7 corr':>9} {'Δ vs base':>10} "
            f"{'bias_med':>9} {'med|Δ|%':>9} {'p99|Δ|%':>9}"
        )
        results: list[tuple[str, dict[str, float]]] = []
        baseline_corr: float | None = None
        for value in values:
            m = compute_ac7_with_seymour(cached, value, mode=mode)
            if value == 0.0:
                baseline_corr = m["corr"]
            delta = (
                (m["corr"] - baseline_corr)
                if baseline_corr is not None
                else float("nan")
            )
            m["delta_vs_base"] = delta
            results.append((mode, m))
            print(
                f"{m['alpha']:>8.4f} {m['n']:>9d} {m['corr']:>+9.4f} "
                f"{delta:>+10.4f} {m['bias_median']:>9.4f} "
                f"{m['median_abs_rel_err_pct']:>9.2f} {m['p99_abs_rel_err_pct']:>9.2f}"
            )
        return results

    results: list[tuple[str, dict[str, float]]] = []
    if args.mode in ("linear", "both"):
        results.extend(_sweep_one(alphas, "linear"))
    if args.mode in ("log", "both"):
        results.extend(_sweep_one(ks, "log"))

    # Best (max corr) — picked across whichever sweeps ran, baseline excluded.
    feasible = [
        (mode, r) for (mode, r) in results
        if np.isfinite(r["corr"]) and r["alpha"] > 0
    ]
    best: tuple[str, dict[str, float]] | None = (
        max(feasible, key=lambda mr: mr[1]["corr"]) if feasible else None
    )
    if best:
        best_mode, best_row = best
        label = "α" if best_mode == "linear" else "k"
        print(
            f"\nBest : mode={best_mode} {label}={best_row['alpha']:.4f} "
            f"(AC7 corr {best_row['corr']:+.4f}, "
            f"bias {best_row['bias_median']:.4f}, "
            f"med|Δ|% {best_row['median_abs_rel_err_pct']:.2f})"
        )

    # Markdown report.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 10 — AC7 Seymour α sweep on QAR cruise cohort",
        "",
        f"> Cohort : {len(flights)} A320 flights, "
        f"{cached['valid_kin'].sum()} valid cruise samples.",
        f"> Cohort age (per flight) : median {np.median(ages_per_flight):.1f} yr, "
        f"mean {ages_per_flight.mean():.2f}, range "
        f"[{ages_per_flight.min():.1f}, {ages_per_flight.max():.1f}].",
        "",
        "## AC7 Pearson corr vs Seymour coefficient",
        "",
        "| mode | coeff | n | AC7 corr | Δ vs base | bias median | median \\|Δ\\| % | p99 \\|Δ\\| % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, r in results:
        lines.append(
            f"| {mode} | {r['alpha']:.4f} | {r['n']} | {r['corr']:+.4f} | "
            f"{r['delta_vs_base']:+.4f} | {r['bias_median']:.4f} | "
            f"{r['median_abs_rel_err_pct']:.2f} | {r['p99_abs_rel_err_pct']:.2f} |"
        )
    if best:
        best_mode, best_row = best
        coeff_label = "α (1/yr)" if best_mode == "linear" else "k (dimensionless)"
        lines += [
            "",
            f"**Best** : mode=`{best_mode}` {coeff_label} = "
            f"`{best_row['alpha']:.4f}` (AC7 corr `{best_row['corr']:+.4f}`, "
            f"Δ vs baseline `{best_row['delta_vs_base']:+.4f}`).",
        ]
    args.report.write_text("\n".join(lines) + "\n")
    print(f"\nMarkdown report → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
