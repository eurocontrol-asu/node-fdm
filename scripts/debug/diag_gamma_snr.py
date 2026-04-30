"""Quantify the SNR of ``gamma`` and ``gamma_diff`` on the A320 ADS-B
training set used by ``node_adsb_v1_A320``.

Goal
----
ADS-B altitude is quantized at 25 ft and resampled near 1 Hz. The
Neural ODE A320 model uses ``gamma_diff = gamma_target - gamma`` as
input, where ``gamma`` is computed from altitude time-derivatives. If
the quantization noise on ``gamma`` dominates the signal in
``gamma_diff``, then any reasoning about the NN sensitivity to
``gamma_diff`` is fragile.

This script measures, on the *training* set:

1. The quantization step actually present in ``raw_alt_ft`` /
   ``raw_alt_m``.
2. The empirical effective sample rate (median ``dt`` per flight).
3. The theoretical and empirical noise on ``gamma``.
4. The signal-to-noise ratio of ``gamma_diff`` per regime (cruise,
   shallow climb / shallow descent, saturated climb / descent).
5. The internal consistency between
   ``gamma_target - gamma == gamma_diff`` (when
   ``gamma_target`` is known).
6. The distribution of effective ``dt`` (gaps).

Output
------
Markdown to stdout (default) or to ``--output``.

Read-only on the Delta table at ``data/flights.delta``.

Usage
-----
    uv run python scripts/debug/diag_gamma_snr.py
    uv run python scripts/debug/diag_gamma_snr.py --output data/mardown/gamma_snr_analysis.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import polars as pl

__all__ = ["main", "compute_snr"]

# --------------------------------------------------------------------------
# Regime thresholds — same as scripts/debug/dataset_regime_stats.py.
# --------------------------------------------------------------------------
GAMMA_CRUISE_RAD = math.radians(0.5)
GAMMA_CLIMB_RAD = math.radians(1.0)
GAMMA_DESCENT_RAD = math.radians(-1.0)

FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS
ALT_RATE_CLIMB_SAT = 1500 * FTMIN_TO_MS
ALT_RATE_DESCENT_SAT = -1000 * FTMIN_TO_MS

# Physical reference: ADS-B altitude quantum (Mode-S / standard ADS-B).
ADSB_ALT_QUANTUM_FT = 25.0
FT_TO_M = 0.3048
ADSB_ALT_QUANTUM_M = ADSB_ALT_QUANTUM_FT * FT_TO_M  # ~7.62 m
TYPICAL_TAS_MS = 230.0  # cruise A320, used only for theoretical estimate.


# --------------------------------------------------------------------------
# Regime classification (re-used).
# --------------------------------------------------------------------------
def _classify_expr() -> pl.Expr:
    gamma = pl.col("fdm_gamma_rad")
    alt_rate = pl.col("fdm_d_alt_ms")
    is_cruise = (gamma.abs() < GAMMA_CRUISE_RAD) & (alt_rate.abs() < ALT_RATE_CRUISE)
    is_climb = (gamma > GAMMA_CLIMB_RAD) | (alt_rate > ALT_RATE_CLIMB)
    is_descent = (gamma < GAMMA_DESCENT_RAD) | (alt_rate < ALT_RATE_DESCENT)
    return (
        pl.when(is_cruise).then(pl.lit("cruise"))
        .when(is_climb).then(pl.lit("climb"))
        .when(is_descent).then(pl.lit("descent"))
        .otherwise(pl.lit("transition"))
        .alias("regime")
    )


def _saturation_expr() -> pl.Expr:
    alt_rate = pl.col("fdm_d_alt_ms")
    return (
        pl.when(alt_rate > ALT_RATE_CLIMB_SAT).then(pl.lit("climb_saturated"))
        .when(alt_rate < ALT_RATE_DESCENT_SAT).then(pl.lit("descent_saturated"))
        .otherwise(pl.lit("shallow"))
        .alias("saturation")
    )


def _phase_expr() -> pl.Expr:
    """Combine regime and saturation into a single 5-class phase label."""
    return (
        pl.when(pl.col("regime") == "cruise").then(pl.lit("cruise"))
        .when((pl.col("regime") == "climb") & (pl.col("saturation") == "climb_saturated"))
        .then(pl.lit("climb_saturated"))
        .when(pl.col("regime") == "climb").then(pl.lit("climb_shallow"))
        .when((pl.col("regime") == "descent") & (pl.col("saturation") == "descent_saturated"))
        .then(pl.lit("descent_saturated"))
        .when(pl.col("regime") == "descent").then(pl.lit("descent_shallow"))
        .otherwise(pl.lit("transition"))
        .alias("phase")
    )


# --------------------------------------------------------------------------
# Core computation.
# --------------------------------------------------------------------------
def compute_snr(
    delta_path: Path,
    *,
    typecode: str = "A320",
    split: str = "train",
) -> dict:
    """Scan the Delta table and gather all SNR-relevant statistics."""
    needed = [
        "meta_aircraft_type",
        "meta_split",
        "meta_flight_id",
        "fdm_flag_valid",
        "raw_timestamp",
        "raw_alt_ft",
        "raw_alt_m",
        "fdm_gamma_rad",
        "fdm_gamma_from_alt_rad",
        "fdm_gamma_target_rad",
        "fdm_gamma_target_known",
        "fdm_gamma_diff_rad",
        "fdm_d_alt_ms",
    ]

    base = (
        pl.scan_delta(str(delta_path))
        .filter(pl.col("fdm_flag_valid"))
        .filter(pl.col("meta_aircraft_type") == typecode)
        .filter(pl.col("meta_split") == split)
        .select(needed)
        .drop_nulls(["fdm_gamma_rad", "fdm_d_alt_ms"])
        .filter(~pl.col("fdm_gamma_rad").is_nan() & ~pl.col("fdm_d_alt_ms").is_nan())
        .with_columns([_classify_expr(), _saturation_expr()])
        .with_columns([_phase_expr()])
        .sort(["meta_flight_id", "raw_timestamp"])
    )

    df = base.collect()
    n_total = df.height
    if n_total == 0:
        return {"error": "no rows after filtering", "delta_path": str(delta_path)}

    # ---- 1. Quantization step in raw_alt_ft -----------------------------
    # Compute per-row delta-altitude (in feet) within each flight.
    df = df.with_columns(
        [
            (pl.col("raw_alt_ft").diff().over("meta_flight_id")).alias("dalt_ft"),
            (pl.col("raw_timestamp").diff().over("meta_flight_id"))
            .dt.total_milliseconds().truediv(1000.0).alias("dt_s"),
        ]
    )

    # Distribution of |dalt_ft| (positive only) — should cluster on multiples of 25 ft.
    dalt_nonzero = df.filter(
        ~pl.col("dalt_ft").is_null() & ~pl.col("dalt_ft").is_nan()
    )["dalt_ft"]
    abs_dalt = dalt_nonzero.abs()
    # Smallest positive |dalt|.
    pos_abs = abs_dalt.filter(abs_dalt > 0)
    quantum_stats = {
        "n_dalt_obs": int(abs_dalt.len()),
        "frac_dalt_zero": float((abs_dalt == 0).mean()) if abs_dalt.len() else math.nan,
        "min_pos_dalt_ft": float(pos_abs.min()) if pos_abs.len() else math.nan,
        "p05_pos_dalt_ft": float(pos_abs.quantile(0.05)) if pos_abs.len() else math.nan,
        "p50_pos_dalt_ft": float(pos_abs.quantile(0.50)) if pos_abs.len() else math.nan,
        # Modulo-25 residual: 0 if all multiples of 25 ft.
        "frac_multiple_of_25ft": float(
            ((abs_dalt % ADSB_ALT_QUANTUM_FT).abs() < 1e-3).mean()
        ) if abs_dalt.len() else math.nan,
    }

    # ---- 2. Sample rate (dt) --------------------------------------------
    dt_series = df.filter(~pl.col("dt_s").is_null() & ~pl.col("dt_s").is_nan())["dt_s"]
    dt_series = dt_series.filter(dt_series > 0)  # drop zero-dt edge cases
    dt_stats = {
        "n_dt_obs": int(dt_series.len()),
        "dt_mean_s": float(dt_series.mean()) if dt_series.len() else math.nan,
        "dt_median_s": float(dt_series.median()) if dt_series.len() else math.nan,
        "dt_p05_s": float(dt_series.quantile(0.05)) if dt_series.len() else math.nan,
        "dt_p95_s": float(dt_series.quantile(0.95)) if dt_series.len() else math.nan,
        "dt_p99_s": float(dt_series.quantile(0.99)) if dt_series.len() else math.nan,
        "dt_max_s": float(dt_series.max()) if dt_series.len() else math.nan,
        "frac_dt_le_2s": float((dt_series <= 2.0).mean()) if dt_series.len() else math.nan,
        "frac_dt_le_5s": float((dt_series <= 5.0).mean()) if dt_series.len() else math.nan,
        "frac_dt_le_10s": float((dt_series <= 10.0).mean()) if dt_series.len() else math.nan,
        "frac_dt_gt_10s": float((dt_series > 10.0).mean()) if dt_series.len() else math.nan,
    }

    # ---- 3. Empirical sigma_gamma per phase -----------------------------
    per_phase = (
        df.group_by("phase")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("fdm_gamma_rad").std().alias("gamma_std"),
                pl.col("fdm_gamma_rad").mean().alias("gamma_mean"),
                pl.col("fdm_gamma_rad").abs().mean().alias("gamma_abs_mean"),
            ]
        )
        .sort("phase")
    )

    # ---- 4. Signal vs noise on gamma_diff per phase ---------------------
    # Filter to rows where gamma_target is known (gamma_diff well-defined).
    gd_ok = ~pl.col("fdm_gamma_diff_rad").is_nan() & (pl.col("fdm_gamma_target_known") > 0.5)
    per_phase_gd = (
        df.filter(gd_ok)
        .group_by("phase")
        .agg(
            [
                pl.len().alias("n_gd_known"),
                pl.col("fdm_gamma_diff_rad").abs().mean().alias("gd_abs_mean"),
                pl.col("fdm_gamma_diff_rad").abs().median().alias("gd_abs_median"),
                pl.col("fdm_gamma_diff_rad").abs().quantile(0.95).alias("gd_abs_p95"),
                pl.col("fdm_gamma_diff_rad").std().alias("gd_std"),
            ]
        )
        .sort("phase")
    )

    # Merge per_phase and per_phase_gd
    merged = per_phase.join(per_phase_gd, on="phase", how="left")

    # ---- 5. Internal consistency: gamma_target - gamma vs gamma_diff ----
    cons = df.filter(
        (~pl.col("fdm_gamma_target_rad").is_null())
        & (~pl.col("fdm_gamma_target_rad").is_nan())
        & (~pl.col("fdm_gamma_diff_rad").is_nan())
        & (pl.col("fdm_gamma_target_known") > 0.5)
    ).select(
        [
            (
                pl.col("fdm_gamma_target_rad")
                - pl.col("fdm_gamma_rad")
                - pl.col("fdm_gamma_diff_rad")
            ).abs().alias("residual"),
        ]
    )
    if cons.height > 0:
        residuals = cons["residual"]
        consistency_stats = {
            "n": cons.height,
            "max_abs_resid": float(residuals.max()),
            "mean_abs_resid": float(residuals.mean()),
            "p99_abs_resid": float(residuals.quantile(0.99)),
        }
    else:
        consistency_stats = {"n": 0}

    # ---- 6. Theoretical noise (depending on chosen dt window) -----------
    # gamma ~ atan(d_alt / (V * dt)). For small angles, sigma_gamma ~
    # sigma_alt / (V * dt). With sigma_alt = 25 ft / sqrt(12) (uniform on
    # quantum), and a measurement involving two altitude reads (start/end),
    # the noise on the difference is sqrt(2) * sigma_alt.
    theoretical = {
        "alt_quantum_ft": ADSB_ALT_QUANTUM_FT,
        "alt_quantum_m": ADSB_ALT_QUANTUM_M,
        "sigma_alt_uniform_m": ADSB_ALT_QUANTUM_M / math.sqrt(12),
        "tas_typical_ms": TYPICAL_TAS_MS,
    }
    # Compute sigma_gamma_theoretical for several dt windows.
    sigma_alt_diff_m = math.sqrt(2.0) * (ADSB_ALT_QUANTUM_M / math.sqrt(12))
    theo_per_dt = []
    for dt in (1.0, 2.0, 4.0, 8.0):
        sigma_gamma = sigma_alt_diff_m / (TYPICAL_TAS_MS * dt)
        theo_per_dt.append(
            {
                "dt_s": dt,
                "sigma_gamma_rad": sigma_gamma,
                "sigma_gamma_deg": math.degrees(sigma_gamma),
            }
        )
    theoretical["per_dt"] = theo_per_dt

    # Cruise-only sigma_gamma (the gold standard for noise calibration).
    cruise_gamma = df.filter(pl.col("phase") == "cruise")["fdm_gamma_rad"]
    cruise_stats = {
        "n": int(cruise_gamma.len()),
        "std": float(cruise_gamma.std()) if cruise_gamma.len() else math.nan,
        "p95_abs": float(cruise_gamma.abs().quantile(0.95)) if cruise_gamma.len() else math.nan,
    }

    return {
        "delta_path": str(delta_path),
        "typecode": typecode,
        "split": split,
        "n_total": n_total,
        "n_flights": int(df["meta_flight_id"].n_unique()),
        "quantum_stats": quantum_stats,
        "dt_stats": dt_stats,
        "per_phase": merged.to_dicts(),
        "consistency": consistency_stats,
        "theoretical": theoretical,
        "cruise_gamma": cruise_stats,
    }


# --------------------------------------------------------------------------
# Markdown rendering.
# --------------------------------------------------------------------------
PHASE_ORDER = [
    "cruise",
    "climb_shallow",
    "climb_saturated",
    "descent_shallow",
    "descent_saturated",
    "transition",
]


def _fmt(v: float | None, fmt: str = ".4f", deg: bool = False, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    if pct:
        return f"{100*v:.2f}%"
    if deg:
        v = math.degrees(v)
    return format(v, fmt)


def render_markdown(stats: dict) -> str:  # noqa: C901 - report formatter
    if "error" in stats:
        return f"# gamma SNR — ERROR\n\n{stats}\n"

    n_total = stats["n_total"]
    n_flights = stats["n_flights"]
    qs = stats["quantum_stats"]
    dts = stats["dt_stats"]
    theo = stats["theoretical"]
    cruise = stats["cruise_gamma"]
    cons = stats["consistency"]

    lines: list[str] = []
    lines.append("# Analyse SNR de `gamma` et `gamma_diff` — A320 train")
    lines.append("")
    lines.append(f"- Delta : `{stats['delta_path']}`")
    lines.append(
        f"- Filtre : `fdm_flag_valid` AND `meta_aircraft_type == '{stats['typecode']}'` "
        f"AND `meta_split == '{stats['split']}'`"
    )
    lines.append(f"- Echantillons : **{n_total:,}** lignes")
    lines.append(f"- Trajectoires : **{n_flights:,}**")
    lines.append("")

    # 1. Colonnes identifiees
    lines.append("## 1. Colonnes identifiees")
    lines.append("")
    lines.append("| Colonne                     | Signification                              | Notes                                  |")
    lines.append("|---|---|---|")
    lines.append("| `raw_timestamp`             | Datetime UTC du sample                      | source des `dt` empiriques              |")
    lines.append("| `raw_alt_ft`                | Altitude ADS-B brute (ft)                   | quantifiee a 25 ft                      |")
    lines.append("| `raw_alt_m`                 | Idem en metres                              | quantum ~7.62 m                         |")
    lines.append("| `fdm_gamma_rad`             | Pente courante (rad)                        | feature input du NN                     |")
    lines.append("| `fdm_gamma_from_alt_rad`    | Pente derivee directement de l'altitude     | reference brute                         |")
    lines.append("| `fdm_gamma_target_rad`      | Consigne de pente FCU/FMS                   | NaN quand inconnue                      |")
    lines.append("| `fdm_gamma_target_known`    | Flag (0/1) target connue                    |                                        |")
    lines.append("| `fdm_gamma_diff_rad`        | `gamma_target - gamma`                      | feature input du NN                     |")
    lines.append("| `fdm_d_alt_ms`              | Vitesse verticale (m/s)                     | utilisee pour stratification regime     |")
    lines.append("")

    # 2. Bruit theorique
    lines.append("## 2. Bruit de quantification theorique")
    lines.append("")
    lines.append(
        f"Quantum ADS-B : **{theo['alt_quantum_ft']:.0f} ft = "
        f"{theo['alt_quantum_m']:.3f} m**.  "
        f"Sigma uniforme sur le quantum : `q/sqrt(12) = "
        f"{theo['sigma_alt_uniform_m']:.3f} m`."
    )
    lines.append("")
    lines.append(
        f"Pour `gamma ~ d(alt)/(V*dt)` avec V = {theo['tas_typical_ms']:.0f} m/s "
        "et bruit independant sur les deux altitudes (start/end) :"
    )
    lines.append("")
    lines.append("`sigma_gamma_theo = sqrt(2) * (q/sqrt(12)) / (V * dt)`")
    lines.append("")
    lines.append("| dt (s) | sigma_gamma (rad) | sigma_gamma (deg) |")
    lines.append("|---:|---:|---:|")
    for r in theo["per_dt"]:
        lines.append(
            f"| {r['dt_s']:.1f} | {r['sigma_gamma_rad']:.4e} | {r['sigma_gamma_deg']:.4f} |"
        )
    lines.append("")

    # 3. Quantum empirique + dt empirique
    lines.append("## 3. Verification empirique du quantum et du sample rate")
    lines.append("")
    lines.append("### 3.a Quantum d'altitude observe")
    lines.append("")
    lines.append(f"- N samples avec `dalt_ft` defini : {qs['n_dalt_obs']:,}")
    lines.append(f"- Fraction `dalt = 0` (alt inchangee) : {_fmt(qs['frac_dalt_zero'], pct=True)}")
    lines.append(f"- min `|dalt|>0` (ft) : {_fmt(qs['min_pos_dalt_ft'], '.3f')}")
    lines.append(f"- p05 `|dalt|>0` (ft) : {_fmt(qs['p05_pos_dalt_ft'], '.3f')}")
    lines.append(f"- mediane `|dalt|>0` (ft) : {_fmt(qs['p50_pos_dalt_ft'], '.3f')}")
    lines.append(
        f"- fraction `|dalt|` multiple de 25 ft : "
        f"{_fmt(qs['frac_multiple_of_25ft'], pct=True)}"
    )
    lines.append("")
    lines.append("### 3.b Sample rate effectif (dt par sample, par vol)")
    lines.append("")
    lines.append(f"- N `dt` observes : {dts['n_dt_obs']:,}")
    lines.append(f"- moyenne dt : {_fmt(dts['dt_mean_s'], '.3f')} s")
    lines.append(f"- mediane dt : {_fmt(dts['dt_median_s'], '.3f')} s")
    lines.append(f"- p05 dt : {_fmt(dts['dt_p05_s'], '.3f')} s")
    lines.append(f"- p95 dt : {_fmt(dts['dt_p95_s'], '.3f')} s")
    lines.append(f"- p99 dt : {_fmt(dts['dt_p99_s'], '.3f')} s")
    lines.append(f"- max dt : {_fmt(dts['dt_max_s'], '.3f')} s")
    lines.append(f"- fraction dt ≤ 2 s : {_fmt(dts['frac_dt_le_2s'], pct=True)}")
    lines.append(f"- fraction dt ≤ 5 s : {_fmt(dts['frac_dt_le_5s'], pct=True)}")
    lines.append(f"- fraction dt ≤ 10 s : {_fmt(dts['frac_dt_le_10s'], pct=True)}")
    lines.append(f"- fraction dt > 10 s : {_fmt(dts['frac_dt_gt_10s'], pct=True)}")
    lines.append("")

    # 4. Bruit empirique sur gamma + SNR de gamma_diff
    lines.append("## 4. Bruit empirique sur `gamma` et SNR de `gamma_diff` par regime")
    lines.append("")
    lines.append(
        "`sigma_gamma` = ecart-type empirique de `gamma` dans le regime. En cruise, "
        "le signal vrai est ~0 -- l'ecart-type est domine par le bruit residuel "
        "(quantification altitude + lissage)."
    )
    lines.append("")
    lines.append(
        "`SNR_gd = mean|gamma_diff| / (sqrt(2) * sigma_gamma_cruise)`. Le facteur "
        "`sqrt(2)` reflete la combinaison du bruit sur `gamma` et sur "
        "`gamma_target` (suppose ~meme amplitude si target est elle-meme lissee)."
    )
    lines.append("")
    sigma_cruise = cruise.get("std")
    if sigma_cruise is None or (isinstance(sigma_cruise, float) and math.isnan(sigma_cruise)):
        sigma_cruise = math.nan
    noise_floor = math.sqrt(2.0) * sigma_cruise if not math.isnan(sigma_cruise) else math.nan
    lines.append(
        f"Reference bruit (cruise) : `sigma_gamma = "
        f"{_fmt(sigma_cruise, '.4e')} rad = "
        f"{_fmt(sigma_cruise, '.4f', deg=True)} deg` "
        f"(N = {cruise['n']:,}). Floor `gamma_diff` "
        f"≈ `{_fmt(noise_floor, '.4e')} rad = "
        f"{_fmt(noise_floor, '.4f', deg=True)} deg`."
    )
    lines.append("")
    lines.append(
        "| Phase             | n         | sigma(gamma) deg | mean\\|gamma\\| deg | "
        "n_gd_known | mean\\|gamma_diff\\| deg | p95\\|gamma_diff\\| deg | SNR_gd |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    by_phase = {row["phase"]: row for row in stats["per_phase"]}
    for phase in PHASE_ORDER:
        row = by_phase.get(phase)
        if row is None:
            lines.append(f"| {phase:<17} | 0 | n/a | n/a | 0 | n/a | n/a | n/a |")
            continue
        gd_mean = row.get("gd_abs_mean")
        gd_p95 = row.get("gd_abs_p95")
        snr = (gd_mean / noise_floor) if (gd_mean is not None and noise_floor and not math.isnan(noise_floor) and noise_floor > 0) else math.nan
        lines.append(
            f"| {phase:<17} | {row['n']:>9,} | "
            f"{_fmt(row.get('gamma_std'), '.4f', deg=True)} | "
            f"{_fmt(row.get('gamma_abs_mean'), '.4f', deg=True)} | "
            f"{int(row.get('n_gd_known') or 0):>10,} | "
            f"{_fmt(gd_mean, '.4f', deg=True)} | "
            f"{_fmt(gd_p95, '.4f', deg=True)} | "
            f"{_fmt(snr, '.2f')} |"
        )
    lines.append("")

    # 5. Coherence interne
    lines.append("## 5. Coherence interne `gamma_target - gamma == gamma_diff`")
    lines.append("")
    if cons.get("n", 0) == 0:
        lines.append("Aucune ligne avec `gamma_target` defini : impossible de verifier.")
    else:
        lines.append(f"- N lignes verifiables : {cons['n']:,}")
        lines.append(
            f"- residual max abs : {_fmt(cons['max_abs_resid'], '.4e')} rad"
        )
        lines.append(
            f"- residual mean abs : {_fmt(cons['mean_abs_resid'], '.4e')} rad"
        )
        lines.append(
            f"- residual p99 abs : {_fmt(cons['p99_abs_resid'], '.4e')} rad"
        )
    lines.append("")

    # 6. Verdict
    lines.append("## 6. Verdict")
    lines.append("")
    cruise_row = by_phase.get("cruise") or {}
    cruise_gd = cruise_row.get("gd_abs_mean")
    cs_row = by_phase.get("climb_saturated") or {}
    ds_row = by_phase.get("descent_saturated") or {}
    csh_row = by_phase.get("climb_shallow") or {}
    dsh_row = by_phase.get("descent_shallow") or {}

    def _snr(row: dict) -> float:
        gd = row.get("gd_abs_mean")
        if gd is None or math.isnan(noise_floor) or noise_floor <= 0:
            return math.nan
        return gd / noise_floor

    snr_cruise = _snr(cruise_row)
    snr_csh = _snr(csh_row)
    snr_cs = _snr(cs_row)
    snr_dsh = _snr(dsh_row)
    snr_ds = _snr(ds_row)

    lines.append(
        f"Reference bruit (cruise) `sigma_gamma = {_fmt(sigma_cruise, '.4f', deg=True)} deg` -- "
        f"floor `gamma_diff` `~= {_fmt(noise_floor, '.4f', deg=True)} deg`."
    )
    lines.append("")
    lines.append("- **cruise** : `mean|gamma_diff| = " + _fmt(cruise_gd, '.4f', deg=True)
                 + f" deg`, SNR = {_fmt(snr_cruise, '.2f')}.")
    lines.append("- **climb_shallow** : SNR = " + _fmt(snr_csh, '.2f') + ".")
    lines.append("- **climb_saturated** : SNR = " + _fmt(snr_cs, '.2f') + ".")
    lines.append("- **descent_shallow** : SNR = " + _fmt(snr_dsh, '.2f') + ".")
    lines.append("- **descent_saturated** : SNR = " + _fmt(snr_ds, '.2f') + ".")
    lines.append("")
    lines.append(
        "Convention de lecture : SNR < 1 -> bruit domine, SNR ~1 -> ambigu, "
        "SNR > 3 -> signal fiable, SNR > 10 -> signal tres net."
    )
    lines.append("")

    # 7. Implications
    lines.append("## 7. Implications")
    lines.append("")
    if not math.isnan(snr_cruise) and snr_cruise < 1.0:
        lines.append(
            "- En cruise, `gamma_diff` est **domine par le bruit de quantification**. "
            "Toute conclusion sur la sensibilite du NN a `gamma_diff` en cruise "
            "est fragile : la perturbation adversariale travaille dans la zone "
            "de bruit, pas dans la zone du signal."
        )
    elif not math.isnan(snr_cruise) and snr_cruise < 3.0:
        lines.append(
            "- En cruise, `gamma_diff` a un SNR marginal (entre 1 et 3). Les "
            "conclusions sur la sensibilite du NN restent fragiles."
        )
    else:
        lines.append(
            "- En cruise, `gamma_diff` semble suffisamment au-dessus du bruit "
            f"(SNR = {_fmt(snr_cruise, '.2f')}). Conclusions exploitables avec prudence."
        )
    if not math.isnan(snr_cs):
        if snr_cs >= 3.0:
            lines.append(
                "- En climb sature, le signal `gamma_diff` est nettement au-dessus "
                "du bruit ; c'est le regime le plus fiable pour evaluer la "
                "sensibilite a `gamma_diff`."
            )
        elif snr_cs >= 1.0:
            lines.append(
                f"- En climb sature, SNR = {_fmt(snr_cs, '.2f')} (proche de 1) : "
                "signal et bruit du meme ordre. Conclusions exploitables mais "
                "avec une incertitude tangible."
            )
        else:
            lines.append(
                f"- En climb sature, SNR = {_fmt(snr_cs, '.2f')} (< 1) : "
                "le bruit domine egalement. Suspect : verifier que le lissage "
                "interne de `gamma_target` n'a pas ecrase le signal."
            )
    if not math.isnan(snr_ds):
        if snr_ds >= 3.0:
            lines.append(
                "- En descent sature, idem : signal robuste."
            )
        elif snr_ds >= 1.0:
            lines.append(
                f"- En descent sature, SNR = {_fmt(snr_ds, '.2f')} (proche de 1) : "
                "ambigu, comme en climb sature."
            )
        else:
            lines.append(
                f"- En descent sature, SNR = {_fmt(snr_ds, '.2f')} (< 1) : "
                "bruit dominant."
            )

    # Note importante sur le lissage interne
    if not math.isnan(sigma_cruise) and sigma_cruise > 0:
        # sigma cruise observed vs theoretical at dt = median
        dt_med = dts.get("dt_median_s") or 4.0
        sigma_alt_diff_m = math.sqrt(2.0) * (ADSB_ALT_QUANTUM_M / math.sqrt(12))
        sigma_theo = sigma_alt_diff_m / (TYPICAL_TAS_MS * dt_med)
        ratio = sigma_cruise / sigma_theo if sigma_theo > 0 else math.nan
        lines.append("")
        lines.append(
            f"- **Note lissage** : sigma(gamma) cruise empirique "
            f"({_fmt(sigma_cruise, '.4f', deg=True)} deg) "
            f"est {_fmt(ratio, '.2f')}x le bruit theorique a dt = "
            f"{dt_med:.1f} s ({_fmt(sigma_theo, '.4f', deg=True)} deg). "
            "Ratio < 1 => `fdm_gamma_rad` est probablement lisse "
            "(Kalman / moyenne glissante / spline) en amont de la table. "
            "Le bruit residuel apres lissage reste neanmoins superieur au signal "
            "moyen `|gamma_diff|` en cruise -- la conclusion qualitative tient."
        )
    lines.append(
        "- Recommandation : conditionner toute analyse de sensibilite NN par "
        "regime, et reporter les SNR par phase. Une analyse globale moyennee "
        "donne un poids excessif au cruise (qui domine numeriquement le dataset) "
        "alors que le SNR y est faible."
    )
    lines.append("")

    # 8. Anomalies / blocages
    lines.append("## 8. Anomalies / blocages")
    lines.append("")
    anomalies = []
    if qs.get("frac_multiple_of_25ft", 0) < 0.95:
        anomalies.append(
            f"`raw_alt_ft` n'est pas strictement quantifie a 25 ft "
            f"({_fmt(qs['frac_multiple_of_25ft'], pct=True)} multiples) -- "
            "verifier si un lissage a deja ete applique."
        )
    if dts.get("dt_p99_s", 0) > 8.0:
        anomalies.append(
            f"Queue de `dt` >8 s observee (p99 = {_fmt(dts['dt_p99_s'], '.2f')} s, "
            f"max = {_fmt(dts['dt_max_s'], '.2f')} s). Sur ces points, le bruit "
            "theorique est plus faible (denominateur V*dt plus grand) MAIS "
            "l'hypothese `pente lineaire` se degrade."
        )
    if cons.get("n", 0) > 0 and cons.get("max_abs_resid", 0) > 1e-3:
        anomalies.append(
            f"residual max `gamma_target - gamma - gamma_diff` = "
            f"{_fmt(cons['max_abs_resid'], '.4e')} rad : suspect, "
            "verifier la chaine de calcul de `gamma_diff`."
        )
    if not anomalies:
        lines.append("Aucune anomalie majeure detectee.")
    else:
        for a in anomalies:
            lines.append(f"- {a}")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
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

    stats = compute_snr(args.delta, typecode=args.typecode, split=args.split)
    report = render_markdown(stats)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
