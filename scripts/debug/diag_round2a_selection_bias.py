"""Round 2A — Selection bias on features (descent_shallow, known=0 vs known=1).

Falsifier
---------
Statistic   : per-feature Cohen's d and 2-sample KS-test between
              `tas_known=0` and `tas_known=1` samples in descent_shallow,
              aggregated over 5 val flights.
Features    : fdm_gamma_rad, fdm_d_alt_ms, fdm_d_tas_ms2, era_tas_ms,
              raw_alt_m, wind_norm (sqrt(u^2+v^2)), position_fraction.
Threshold   : H_2A SURVIVES if at least one feature has |Cohen's d| > 0.5
              AND KS p < 0.01.
              H_2A DIES if all features have |Cohen's d| < 0.3 AND
              all KS p > 0.05.
Null world  : if H_2A were false, all |d| < 0.2 and KS not significant.
Data slice  : val seed 0, 5 flights (3c6634_DLH4PV_s0, 0aca66_AVA8443_s0,
              0aca66_AVA8447_s0, 0aca66_AVA8548_s0, 0aca66_AVA8558_s0),
              descent_shallow only.
Strata      : per-flight + aggregated (with and without AVA8447).

Usage
-----
    uv run python scripts/debug/diag_round2a_selection_bias.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

__all__ = ["main"]

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DELTA_PATH = Path("data/flights.delta")
REPORT_PATH = Path("data/mardown/round2a_selection_bias.md")

FLIGHTS = [
    "3c6634_DLH4PV_s0",
    "0aca66_AVA8443_s0",
    "0aca66_AVA8447_s0",
    "0aca66_AVA8548_s0",
    "0aca66_AVA8558_s0",
]
FLIGHT_AVA8447 = "0aca66_AVA8447_s0"

# Phase thresholds (mirror diag_target_known_stratification.py).
GAMMA_CRUISE_RAD = math.radians(0.5)
GAMMA_CLIMB_RAD = math.radians(1.0)
GAMMA_DESCENT_RAD = math.radians(-1.0)

FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS
ALT_RATE_CLIMB_SAT = 1500 * FTMIN_TO_MS
ALT_RATE_DESCENT_SAT = -1000 * FTMIN_TO_MS

FEATURES = [
    "fdm_gamma_rad",
    "fdm_d_alt_ms",
    "fdm_d_tas_ms2",
    "era_tas_ms",
    "raw_alt_m",
    "wind_norm",
    "position_fraction_in_flight",
]


# ----------------------------------------------------------------------
# Phase classification (matches diag_target_known_stratification)
# ----------------------------------------------------------------------
def _phase_label(gamma: float, d_alt: float) -> str:
    """Return phase label for a single sample."""
    if abs(gamma) < GAMMA_CRUISE_RAD and abs(d_alt) < ALT_RATE_CRUISE:
        return "cruise"
    if gamma > GAMMA_CLIMB_RAD or d_alt > ALT_RATE_CLIMB:
        return "climb_saturated" if d_alt > ALT_RATE_CLIMB_SAT else "climb_shallow"
    if gamma < GAMMA_DESCENT_RAD or d_alt < ALT_RATE_DESCENT:
        return (
            "descent_saturated" if d_alt < ALT_RATE_DESCENT_SAT
            else "descent_shallow"
        )
    return "transition"


def _classify_phases_np(gamma: np.ndarray, d_alt: np.ndarray) -> np.ndarray:
    out = np.empty(len(gamma), dtype=object)
    for i in range(len(gamma)):
        out[i] = _phase_label(float(gamma[i]), float(d_alt[i]))
    return out


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
def load_flight_features(flight_id: str) -> pl.DataFrame:
    """Pull raw rows for a single flight, restrict to descent_shallow."""
    df = (
        pl.scan_delta(str(DELTA_PATH))
        .filter(pl.col("meta_flight_id") == flight_id)
        .filter(pl.col("fdm_flag_valid"))
        .filter(pl.col("meta_split") == "val")
        .select(
            [
                "raw_timestamp",
                "fdm_gamma_rad",
                "fdm_d_alt_ms",
                "fdm_d_tas_ms2",
                "era_tas_ms",
                "raw_alt_m",
                "era_u_wind_ms",
                "era_v_wind_ms",
                "fdm_tas_target_known",
            ]
        )
        .sort("raw_timestamp")
        .collect()
    )
    if df.height == 0:
        return df

    # Compute derived features.
    n = df.height
    pos = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    wind_norm = np.sqrt(
        df["era_u_wind_ms"].to_numpy() ** 2
        + df["era_v_wind_ms"].to_numpy() ** 2
    )
    df = df.with_columns(
        pl.Series("wind_norm", wind_norm),
        pl.Series("position_fraction_in_flight", pos),
        pl.col("fdm_tas_target_known").cast(pl.Float64).alias("tas_known_f"),
    )

    # Phase classification.
    gamma = df["fdm_gamma_rad"].to_numpy()
    d_alt = df["fdm_d_alt_ms"].to_numpy()
    finite = np.isfinite(gamma) & np.isfinite(d_alt)
    df = df.filter(pl.Series(finite))
    if df.height == 0:
        return df
    labels = _classify_phases_np(
        df["fdm_gamma_rad"].to_numpy(), df["fdm_d_alt_ms"].to_numpy()
    )
    df = df.with_columns(pl.Series("phase", labels))
    return df.filter(pl.col("phase") == "descent_shallow")


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def cohens_d(x0: np.ndarray, x1: np.ndarray) -> tuple[float, float]:
    """Cohen's d with pooled std. Returns (d, pooled_std)."""
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return float("nan"), float("nan")
    m0, m1 = float(np.mean(x0)), float(np.mean(x1))
    v0 = float(np.var(x0, ddof=1))
    v1 = float(np.var(x1, ddof=1))
    pooled_var = ((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else float("nan")
    if not math.isfinite(pooled_std) or pooled_std == 0.0:
        return float("nan"), pooled_std
    return (m0 - m1) / pooled_std, pooled_std


def feature_stats(
    df: pl.DataFrame, feature: str
) -> dict:
    """Compute per-feature Cohen's d + KS for known=0 vs known=1."""
    known = df["tas_known_f"].to_numpy()
    vals = df[feature].to_numpy().astype(np.float64)
    finite = np.isfinite(vals) & np.isfinite(known)
    vals = vals[finite]
    known = known[finite]
    x0 = vals[known < 0.5]
    x1 = vals[known >= 0.5]
    if len(x0) < 2 or len(x1) < 2:
        return {
            "feature": feature,
            "n_0": int(len(x0)),
            "n_1": int(len(x1)),
            "mean_0": float(np.mean(x0)) if len(x0) else float("nan"),
            "mean_1": float(np.mean(x1)) if len(x1) else float("nan"),
            "std_pooled": float("nan"),
            "cohens_d": float("nan"),
            "ks_stat": float("nan"),
            "ks_pvalue": float("nan"),
        }
    d, pooled = cohens_d(x0, x1)
    ks = stats.ks_2samp(x0, x1, alternative="two-sided", method="auto")
    return {
        "feature": feature,
        "n_0": int(len(x0)),
        "n_1": int(len(x1)),
        "mean_0": float(np.mean(x0)),
        "mean_1": float(np.mean(x1)),
        "std_pooled": pooled,
        "cohens_d": d,
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def aggregate_table(per_flight: dict[str, pl.DataFrame], flights: list[str]) -> list[dict]:
    """Stack per-flight descent_shallow data and compute stats per feature."""
    if not flights:
        return [
            {**feature_stats(pl.DataFrame(), f), "feature": f} for f in FEATURES
        ]
    sub = pl.concat([per_flight[f] for f in flights if per_flight[f].height > 0])
    if sub.height == 0:
        return [feature_stats(sub, f) for f in FEATURES]
    return [feature_stats(sub, f) for f in FEATURES]


def per_flight_top(rows: list[dict]) -> tuple[str, float, float, int, int]:
    """Pick the feature with the largest |Cohen's d| (finite only)."""
    best_feat = "n/a"
    best_abs_d = -1.0
    best_d = float("nan")
    best_p = float("nan")
    best_n0 = 0
    best_n1 = 0
    for r in rows:
        d = r["cohens_d"]
        if not math.isfinite(d):
            continue
        if abs(d) > best_abs_d:
            best_abs_d = abs(d)
            best_d = d
            best_feat = r["feature"]
            best_p = r["ks_pvalue"]
            best_n0 = r["n_0"]
            best_n1 = r["n_1"]
    return best_feat, best_d, best_p, best_n0, best_n1


# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------
def verdict(agg_rows: list[dict]) -> tuple[str, str]:
    """Return (verdict_label, sentence)."""
    finite = [r for r in agg_rows if math.isfinite(r["cohens_d"])]
    if not finite:
        return "indeterminate", "No finite Cohen's d available across features."
    max_row = max(finite, key=lambda r: abs(r["cohens_d"]))
    max_abs_d = abs(max_row["cohens_d"])
    max_feat = max_row["feature"]
    ks_p = max_row["ks_pvalue"]

    survives = max_abs_d > 0.5 and (math.isfinite(ks_p) and ks_p < 0.01)
    dies = all(
        abs(r["cohens_d"]) < 0.3 for r in finite
    ) and all(
        (not math.isfinite(r["ks_pvalue"])) or r["ks_pvalue"] > 0.05
        for r in finite
    )
    if survives:
        label = "survives"
    elif dies:
        label = "dies"
    else:
        label = "indeterminate"
    sentence = (
        f"H_2A {label}: max |Cohen's d| = {max_abs_d:.3f} on feature `{max_feat}` "
        f"(KS p = {ks_p:.3g})."
    )
    return label, sentence


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def _f(v: float, fmt: str = ".4f") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return format(v, fmt)


def _pval(v: float) -> str:
    if v is None or math.isnan(v):
        return "n/a"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def render_table(rows: list[dict]) -> list[str]:
    out = [
        "| Feature | n_0 | n_1 | mean_0 | mean_1 | std_pooled | Cohen's d | KS stat | KS p-value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            f"| `{r['feature']}` | {r['n_0']} | {r['n_1']} | "
            f"{_f(r['mean_0'])} | {_f(r['mean_1'])} | {_f(r['std_pooled'])} | "
            f"{_f(r['cohens_d'], '+.3f')} | {_f(r['ks_stat'])} | "
            f"{_pval(r['ks_pvalue'])} |"
        )
    return out


def render_report(
    agg_rows_all: list[dict],
    agg_rows_no8447: list[dict],
    per_flight_rows: list[tuple[str, list[dict], int, int]],
    per_flight_data: dict[str, pl.DataFrame],
) -> str:
    label, sentence = verdict(agg_rows_all)
    lines: list[str] = []
    lines.append("# Round 2A — Selection bias on features")
    lines.append("")
    lines.append("## Hypothesis under test")
    lines.append("")
    lines.append(
        "Les samples descent_shallow `tas_known=0` sont systématiquement dans des"
        " régions de l'espace des features plus difficiles à prédire que les samples"
        " `tas_known=1` du même régime sur les mêmes vols — différences distributionnelles"
        " sur (γ, alt_rate, tas_rate, tas absolue, vent, position-dans-vol) — et le bias"
        " +19.73 m/s vient de cette difficulté intrinsèque, pas de l'absence de target."
    )
    lines.append("")
    lines.append("## Falsifier")
    lines.append("")
    lines.append(
        "- **Statistic** : pour chaque feature dans"
        " {fdm_gamma_rad, fdm_d_alt_ms, fdm_d_tas_ms2, era_tas_ms, raw_alt_m, wind_norm,"
        " position_fraction_in_flight}, calculer Cohen's d entre les distributions"
        " `tas_known=0` et `tas_known=1` sur descent_shallow agrégé 5 vols."
        " Aussi Kolmogorov-Smirnov 2-sample (`scipy.stats.ks_2samp`) avec p-value."
    )
    lines.append(
        "- **Threshold** : H_2A SURVIVES si AU MOINS UNE feature a `|Cohen's d| > 0.5`"
        " ET `KS p < 0.01`. H_2A DIES si TOUTES les features ont `|d| < 0.3`"
        " ET `KS p > 0.05`."
    )
    lines.append(
        "- **Null world** : si H_2A était fausse (pas de biais de sélection), les"
        " distributions de features seraient indistinguables entre known=0 et known=1"
        " (toutes |d| < 0.2, KS non-significatif)."
    )
    lines.append(
        "- **Data slice** : val seed 0, 5 vols ("
        "3c6634_DLH4PV_s0, 0aca66_AVA8443_s0, 0aca66_AVA8447_s0,"
        " 0aca66_AVA8548_s0, 0aca66_AVA8558_s0), descent_shallow uniquement."
    )
    lines.append(
        "- **Strata** : per-flight (audit) + agrégé sur 5 vols et agrégé sans AVA8447."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Pour chaque vol on lit les samples valides (`fdm_flag_valid AND meta_split == 'val'`)"
        " depuis `data/flights.delta`, on classe les phases avec les seuils"
        " `dataset_regime_stats.py` (γ et alt_rate), on retient `descent_shallow`,"
        " puis on stratifie par `fdm_tas_target_known` (cast Float). On calcule Cohen's d"
        " (pooled std avec ddof=1, `s_p = sqrt(((n0-1)v0+(n1-1)v1)/(n0+n1-2))`) et KS"
        " 2-sample. Les NaN sont filtrés feature-par-feature avant calcul."
        " `wind_norm = sqrt(era_u_wind_ms^2 + era_v_wind_ms^2)`."
        " `position_fraction_in_flight = (i+1)/N` sur la séquence triée par `raw_timestamp`."
    )
    lines.append("")

    lines.append("## Result")
    lines.append("")
    lines.append("### Aggregate (5 flights, descent_shallow, known=0 vs known=1)")
    lines.append("")
    lines.extend(render_table(agg_rows_all))
    lines.append("")
    lines.append("### Aggregate without AVA8447")
    lines.append("")
    lines.extend(render_table(agg_rows_no8447))
    lines.append("")

    lines.append("### Per-flight summary (top feature only)")
    lines.append("")
    lines.append("| Flight | n_0 | n_1 | top_feature | Cohen's d | KS p |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for fid, rows, n0, n1 in per_flight_rows:
        feat, d, p, _, _ = per_flight_top(rows)
        lines.append(
            f"| `{fid}` | {n0} | {n1} | `{feat}` | "
            f"{_f(d, '+.3f')} | {_pval(p)} |"
        )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(sentence)
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- `position_fraction_in_flight` est un proxy `(i+1)/N` calculé sur les rows"
        " valides triées par `raw_timestamp`, indépendamment de la durée physique entre"
        " samples (fixée à 4 s par STEP_S, donc raisonnable). Il ne distingue pas"
        " \"position dans la descente\" de \"position dans tout le vol\" : un vol descendant"
        " tard verra ses samples descent_shallow concentrés dans la moitié haute, ce qui"
        " peut amplifier artificiellement |d| sur cette feature sans que cela reflète"
        " une difficulté de prédiction."
    )
    lines.append(
        "- AVA8447 n'a que `n=6` samples descent_shallow au total ; toute statistique"
        " stratifiée sur ce vol est extrêmement bruitée. La table sans AVA8447 est le"
        " test plus robuste de l'hypothèse."
    )
    lines.append(
        "- `std_pooled` utilise la formule `sqrt(((n0-1)*var0+(n1-1)*var1)/(n0+n1-2))`"
        " avec `ddof=1`. Si une strate est vide ou si pooled_var est 0/non-fini, la"
        " ligne reporte `n/a` pour Cohen's d."
    )
    lines.append(
        "- Les NaN sont filtrés feature-par-feature : `n_0` et `n_1` peuvent varier"
        " légèrement entre features sur un même vol."
    )
    lines.append(
        "- Pas de feature `wind_norm` native dans le schéma : recalculée comme"
        " `sqrt(era_u_wind_ms^2 + era_v_wind_ms^2)`. La variable `fdm_long_wind_ms`"
        " (composante longitudinale signée) existe mais n'est pas utilisée car le"
        " brief demande l'amplitude vent."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("Round 2A — Selection bias on features (descent_shallow, known=0 vs known=1)")
    print("=" * 78)
    print("Falsifier:")
    print("  - Statistic: Cohen's d + KS 2-sample on 7 features")
    print("  - Threshold: SURVIVES if any |d|>0.5 AND KS p<0.01;")
    print("               DIES if all |d|<0.3 AND all KS p>0.05")
    print("  - Null world: indistinguishable distributions (|d|<0.2)")
    print("  - Data slice: val seed 0, 5 flights, descent_shallow only")
    print("  - Strata: per-flight + aggregate (with/without AVA8447)")
    print("=" * 78)

    per_flight_data: dict[str, pl.DataFrame] = {}
    for fid in FLIGHTS:
        df = load_flight_features(fid)
        per_flight_data[fid] = df
        print(f"[load] {fid}: descent_shallow rows = {df.height}")

    # Aggregate stats.
    print("\n[stats] aggregate over 5 flights ...")
    agg_all = aggregate_table(per_flight_data, FLIGHTS)
    no8447 = [f for f in FLIGHTS if f != FLIGHT_AVA8447]
    print("[stats] aggregate without AVA8447 ...")
    agg_no = aggregate_table(per_flight_data, no8447)

    # Per-flight stats.
    per_flight_rows: list[tuple[str, list[dict], int, int]] = []
    for fid in FLIGHTS:
        df = per_flight_data[fid]
        if df.height == 0:
            per_flight_rows.append((fid, [], 0, 0))
            continue
        rows = [feature_stats(df, f) for f in FEATURES]
        n0 = max((r["n_0"] for r in rows), default=0)
        n1 = max((r["n_1"] for r in rows), default=0)
        per_flight_rows.append((fid, rows, n0, n1))
        print(f"[stats] {fid}: n0_max={n0}, n1_max={n1}")

    label, sentence = verdict(agg_all)
    print("\n" + "=" * 78)
    print(f"VERDICT: {sentence}")
    print("=" * 78)

    md = render_report(agg_all, agg_no, per_flight_rows, per_flight_data)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
