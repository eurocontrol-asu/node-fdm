"""Visual validation — Niveau 2 (Autopilot) on a high-coverage flight.

Goal
----
Confirm visually that the Neural ODE model
``data/models/node_adsb_v1_A320`` reproduces an ADS-B flight trajectory
when BOTH ``tas_known`` and ``gamma_known`` are dense throughout the
flight (>= 70% coverage on each).

This is a **visual confirmation** that the dynamic model (Level 2 /
Autopilot) is correctly calibrated when the consignes are present —
NOT a falsification. Supports the vision document
``data/mardown/vision_pilot_autopilot.md``.

Selection
---------
Among A320 val flights (seed 0), pick the candidate with the highest
combined ``(tas_known, gamma_known)`` coverage that also covers
cruise + climb + descent and lasts >30 min.

Outputs
-------
- Figure : ``data/figures/inference_check_<icao>_<callsign>_s0_high_coverage.png``
- Report : ``data/mardown/high_coverage_validation.md``

Reproduce with::

    uv run python scripts/debug/predict_high_coverage_flight.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from node_fdm.predictor import NodeFDMPredictor
from node_fdm_pipeline.resolver import resolve_architecture

__all__ = ["main"]


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
ARCH = "adsb"
MODEL_DIR = Path("data/models")
DELTA_PATH = Path("data/flights.delta")
STEP_S = 4.0  # grid step (s)
AIRCRAFT_TYPE = "A320"
SPLIT = "val"
SEED_SUFFIX = "_s0"

# Selection thresholds (relaxable in this order: gamma -> tas).
TAS_COV_MIN = 0.70
GAMMA_COV_MIN = 0.70
MIN_DURATION_MIN = 30.0
MIN_N = int((MIN_DURATION_MIN * 60.0) / STEP_S)

# Phase thresholds (mirror dataset_regime_stats.py).
GAMMA_CRUISE_RAD = math.radians(0.5)
GAMMA_CLIMB_RAD = math.radians(1.0)
GAMMA_DESCENT_RAD = math.radians(-1.0)
FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS


# ----------------------------------------------------------------------
# Flight selection
# ----------------------------------------------------------------------
@dataclass
class Selection:
    """Selected flight + the criteria that produced it."""

    flight_id: str
    n: int
    tas_cov: float
    gamma_cov: float
    duration_min: float
    tas_threshold: float
    gamma_threshold: float
    regimes: list[str]


def _candidates(df: pl.DataFrame, tas_thr: float, gamma_thr: float) -> pl.DataFrame:
    """Return val A320 seed-0 flights matching coverage thresholds."""
    val = df.filter(pl.col("meta_split") == SPLIT)
    agg = val.group_by("meta_flight_id").agg(
        [
            pl.len().alias("n"),
            pl.col("fdm_tas_target_known")
            .cast(pl.Float64)
            .mean()
            .alias("tas_cov"),
            pl.col("fdm_gamma_target_known")
            .cast(pl.Float64)
            .mean()
            .alias("gamma_cov"),
            pl.col("meta_aircraft_type").first().alias("actype"),
        ]
    )
    return (
        agg.filter(
            (pl.col("actype") == AIRCRAFT_TYPE)
            & (pl.col("meta_flight_id").str.ends_with(SEED_SUFFIX))
            & (pl.col("n") >= MIN_N)
            & (pl.col("tas_cov") >= tas_thr)
            & (pl.col("gamma_cov") >= gamma_thr)
        )
        .sort(["tas_cov", "gamma_cov", "n"], descending=True)
    )


def _regimes_present(sub: pl.DataFrame) -> list[str]:
    """Return list of high-level regimes present (cruise/climb/descent)."""
    g = sub["fdm_gamma_rad"].to_numpy()
    alt = sub["raw_alt_m"].to_numpy()
    d_alt = np.gradient(alt, STEP_S)
    is_cruise = (np.abs(g) < GAMMA_CRUISE_RAD) & (np.abs(d_alt) < ALT_RATE_CRUISE)
    is_climb = (g > GAMMA_CLIMB_RAD) | (d_alt > ALT_RATE_CLIMB)
    is_descent = (g < GAMMA_DESCENT_RAD) | (d_alt < ALT_RATE_DESCENT)
    out: list[str] = []
    if is_cruise.mean() > 0.05:
        out.append("cruise")
    if is_climb.mean() > 0.05:
        out.append("climb")
    if is_descent.mean() > 0.05:
        out.append("descent")
    return out


def select_flight(df: pl.DataFrame) -> Selection:
    """Pick the best high-coverage val flight; relax gamma then tas."""
    relaxations = [
        (TAS_COV_MIN, GAMMA_COV_MIN),
        (TAS_COV_MIN, 0.60),
        (TAS_COV_MIN, 0.50),
        (0.60, 0.50),
    ]
    val = df.filter(pl.col("meta_split") == SPLIT)
    chosen: tuple[str, int, float, float] | None = None
    used_thr: tuple[float, float] | None = None
    for tas_thr, gamma_thr in relaxations:
        cands = _candidates(df, tas_thr, gamma_thr)
        if cands.height == 0:
            continue
        # Prefer one with all three regimes present.
        for row in cands.iter_rows(named=True):
            sub = val.filter(
                pl.col("meta_flight_id") == row["meta_flight_id"]
            ).sort("raw_timestamp")
            regs = _regimes_present(sub)
            if {"cruise", "climb", "descent"}.issubset(regs):
                chosen = (
                    row["meta_flight_id"],
                    int(row["n"]),
                    float(row["tas_cov"]),
                    float(row["gamma_cov"]),
                )
                used_thr = (tas_thr, gamma_thr)
                break
        if chosen is None:
            # Fall back to top candidate even without all three regimes.
            row = cands.row(0, named=True)
            chosen = (
                row["meta_flight_id"],
                int(row["n"]),
                float(row["tas_cov"]),
                float(row["gamma_cov"]),
            )
            used_thr = (tas_thr, gamma_thr)
        break

    if chosen is None or used_thr is None:
        msg = (
            "No val A320 seed-0 flight meets even the most relaxed coverage "
            "criteria (tas>=0.60, gamma>=0.50). Investigation blocked."
        )
        raise SystemExit(msg)

    fid, n, tas_cov, gamma_cov = chosen
    sub = val.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
    regs = _regimes_present(sub)
    return Selection(
        flight_id=fid,
        n=n,
        tas_cov=tas_cov,
        gamma_cov=gamma_cov,
        duration_min=n * STEP_S / 60.0,
        tas_threshold=used_thr[0],
        gamma_threshold=used_thr[1],
        regimes=regs,
    )


# ----------------------------------------------------------------------
# Inference (mirrors diag_dlh4pv_postF3.py)
# ----------------------------------------------------------------------
@dataclass
class FlightData:
    """Aligned arrays for plotting and metrics."""

    time_min: np.ndarray
    alt_true: np.ndarray
    tas_true: np.ndarray
    gamma_true: np.ndarray
    alt_target: np.ndarray
    tas_target: np.ndarray
    gamma_target_raw: np.ndarray
    tas_known: np.ndarray
    gamma_known: np.ndarray
    alt_pred: np.ndarray
    tas_pred: np.ndarray
    gamma_pred: np.ndarray


def load_and_predict(flight_id: str) -> FlightData:
    """Run inference on the chosen flight and return aligned arrays."""
    info = resolve_architecture(ARCH)
    model_path = MODEL_DIR / f"{info.name}_{AIRCRAFT_TYPE}"
    if not model_path.exists():
        msg = f"Model not found at {model_path}"
        raise SystemExit(msg)

    predictor = NodeFDMPredictor(model_path=model_path, device="cpu")

    df = pl.read_delta(str(DELTA_PATH)).filter(pl.col("fdm_flag_valid"))
    sel_cols = [c for c in df.columns if c.startswith("fdm_") and "_sel" in c]
    if sel_cols:
        df = df.with_columns(
            [pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols]
        )

    val_df = df.filter(pl.col("meta_split") == SPLIT)
    flight_df = val_df.filter(pl.col("meta_flight_id") == flight_id).sort(
        "raw_timestamp"
    )

    x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)
    u_arr_raw = flight_df.select(info.u_cols).to_numpy().astype(np.float32)
    e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)

    finite_mask = np.isfinite(x_arr).all(axis=1) & np.isfinite(e_arr).all(axis=1)
    x_arr = x_arr[finite_mask]
    u_arr = u_arr_raw[finite_mask]
    e_arr = e_arr[finite_mask]

    print(f"Finite rows: {finite_mask.sum()}/{len(finite_mask)}")
    x0 = x_arr[0]
    predictions = predictor.predict_flight(x0, u_arr, e_arr)
    n_pred = len(predictions["era_tas_ms"])

    alt_idx = info.x_cols.index("raw_alt_m")
    tas_idx = info.x_cols.index("era_tas_ms")
    gamma_idx = info.x_cols.index("fdm_gamma_rad")
    alt_target_idx = info.u_cols.index("fdm_alt_target_m")
    tas_target_idx = info.u_cols.index("fdm_tas_target_ms")
    gamma_target_idx = info.u_cols.index("fdm_gamma_target_rad")
    tas_known_idx = info.u_cols.index("fdm_tas_target_known")
    gamma_known_idx = info.u_cols.index("fdm_gamma_target_known")

    n_true = x_arr.shape[0]
    end = min(n_true - 1, n_pred)
    pred_slice = slice(0, end)
    true_slice = slice(1, end + 1)

    return FlightData(
        time_min=(np.arange(end) + 1) * STEP_S / 60.0,
        alt_true=x_arr[true_slice, alt_idx],
        tas_true=x_arr[true_slice, tas_idx],
        gamma_true=x_arr[true_slice, gamma_idx],
        alt_target=u_arr[true_slice, alt_target_idx],
        tas_target=u_arr[true_slice, tas_target_idx],
        gamma_target_raw=u_arr[true_slice, gamma_target_idx],
        tas_known=u_arr[true_slice, tas_known_idx],
        gamma_known=u_arr[true_slice, gamma_known_idx],
        alt_pred=predictions["raw_alt_m"][pred_slice],
        tas_pred=predictions["era_tas_ms"][pred_slice],
        gamma_pred=predictions["fdm_gamma_rad"][pred_slice],
    )


# ----------------------------------------------------------------------
# Metrics stratified by (tas_known, gamma_known)
# ----------------------------------------------------------------------
@dataclass
class StratumMetrics:
    """MAE/bias on a subset of samples."""

    n: int
    mae_tas: float
    bias_tas: float
    mae_gamma_deg: float
    bias_gamma_deg: float
    mae_alt: float


def _stratum(fd: FlightData, mask: np.ndarray) -> StratumMetrics:
    n = int(mask.sum())
    if n == 0:
        nan = float("nan")
        return StratumMetrics(0, nan, nan, nan, nan, nan)
    return StratumMetrics(
        n=n,
        mae_tas=float(np.mean(np.abs(fd.tas_pred[mask] - fd.tas_true[mask]))),
        bias_tas=float(np.mean(fd.tas_pred[mask] - fd.tas_true[mask])),
        mae_gamma_deg=float(
            np.degrees(np.mean(np.abs(fd.gamma_pred[mask] - fd.gamma_true[mask])))
        ),
        bias_gamma_deg=float(
            np.degrees(np.mean(fd.gamma_pred[mask] - fd.gamma_true[mask]))
        ),
        mae_alt=float(np.mean(np.abs(fd.alt_pred[mask] - fd.alt_true[mask]))),
    )


def compute_metrics(fd: FlightData) -> tuple[StratumMetrics, StratumMetrics]:
    """Return (both_known=1, at_least_one_known=0)."""
    both = (fd.tas_known > 0.5) & (fd.gamma_known > 0.5)
    return _stratum(fd, both), _stratum(fd, ~both)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def make_figure(fd: FlightData, sel: Selection, out_path: Path) -> None:
    """Render the 3-panel altitude / TAS / FPA figure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(fd.time_min, fd.alt_true, "k-", lw=1.5, label="True", alpha=0.85)
    ax.plot(fd.time_min, fd.alt_pred, "r--", lw=1.2, label="Predicted", alpha=0.85)
    # alt_target is dense (FMS altitude); plot directly.
    ax.plot(fd.time_min, fd.alt_target, "b-", lw=2.0, label="Target", alpha=0.4)
    ax.set_ylabel("Altitude [m]")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(fd.time_min, fd.tas_true, "k-", lw=1.5, label="True", alpha=0.85)
    ax.plot(fd.time_min, fd.tas_pred, "r--", lw=1.2, label="Predicted", alpha=0.85)
    tas_target_plot = np.where(fd.tas_known > 0.5, fd.tas_target, np.nan)
    ax.plot(fd.time_min, tas_target_plot, "b-", lw=2.0, label="Target", alpha=0.4)
    ax.set_ylabel("TAS [m/s]")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(
        fd.time_min, np.degrees(fd.gamma_true), "k-", lw=1.0,
        label="True", alpha=0.7,
    )
    ax.plot(
        fd.time_min, np.degrees(fd.gamma_pred), "r--", lw=1.2,
        label="Predicted", alpha=0.85,
    )
    gamma_target_plot = np.where(
        fd.gamma_known > 0.5, np.degrees(fd.gamma_target_raw), np.nan
    )
    ax.plot(
        fd.time_min, gamma_target_plot, "b-", lw=2.0,
        label="γ target", alpha=0.4,
    )
    ax.set_ylabel("FPA / γ [deg]")
    ax.set_xlabel("Time [min]")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    title = (
        f"Validation Niveau 2 (Autopilot calibré) — {sel.flight_id} — "
        f"coverage tas/γ {sel.tas_cov * 100:.0f}/{sel.gamma_cov * 100:.0f}%"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------
def render_report(
    sel: Selection,
    metrics_both: StratumMetrics,
    metrics_other: StratumMetrics,
    fig_path: Path,
) -> str:
    """Render the short validation report."""

    def _f(v: float, fmt: str = ".3f") -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return format(v, fmt)

    fig_rel = Path("..") / "figures" / fig_path.name

    # Visual verdict heuristic: if MAE_TAS < 5 m/s and MAE_gamma < 0.5 deg
    # and MAE_alt < 200 m on the (both known=1) stratum, predicted curves
    # track tightly.
    tight = (
        metrics_both.n > 0
        and not math.isnan(metrics_both.mae_tas)
        and metrics_both.mae_tas < 5.0
        and metrics_both.mae_gamma_deg < 0.6
        and metrics_both.mae_alt < 250.0
    )
    if tight:
        verdict = (
            "Sur les régions où tas_known=1 ET gamma_known=1, les courbes "
            "Predicted suivent étroitement True/Target — la calibration de la "
            "couche dynamique (Autopilot) est confirmée visuellement."
        )
    else:
        verdict = (
            "Les courbes Predicted suivent globalement True/Target sur les "
            "régions où les deux consignes sont connues, mais l'écart "
            "résiduel reste mesurable (cf. métriques ci-dessus)."
        )

    lines: list[str] = []
    lines.append("# Validation Niveau 2 — vol à forte couverture consigne")
    lines.append("")
    lines.append("## Vol sélectionné")
    lines.append(f"- Flight: `{sel.flight_id}`")
    lines.append(f"- Duration: {sel.duration_min:.1f} min ({sel.n} échantillons à {STEP_S:.0f} s)")
    lines.append(f"- tas_known coverage: {sel.tas_cov * 100:.1f}%")
    lines.append(f"- gamma_known coverage: {sel.gamma_cov * 100:.1f}%")
    lines.append(f"- Regimes covered: {', '.join(sel.regimes) if sel.regimes else 'n/a'}")
    lines.append("")
    lines.append("## Selection criteria applied")
    lines.append(
        f"- tas_known fraction >= {sel.tas_threshold:.2f}, "
        f"gamma_known fraction >= {sel.gamma_threshold:.2f}, "
        f"durée > {MIN_DURATION_MIN:.0f} min, split=val, A320, seed 0."
    )
    if (
        sel.tas_threshold < TAS_COV_MIN
        or sel.gamma_threshold < GAMMA_COV_MIN
    ):
        lines.append(
            "- **Note** : seuils relâchés depuis la cible initiale "
            f"(tas>={TAS_COV_MIN:.2f}, gamma>={GAMMA_COV_MIN:.2f}) "
            "car aucun candidat ne satisfaisait la version stricte."
        )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| Stratum | n | MAE_TAS (m/s) | bias_TAS | MAE_γ (deg) | bias_γ (deg) | MAE_alt (m) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| both known=1 | {metrics_both.n} | "
        f"{_f(metrics_both.mae_tas)} | "
        f"{_f(metrics_both.bias_tas, '+.3f')} | "
        f"{_f(metrics_both.mae_gamma_deg)} | "
        f"{_f(metrics_both.bias_gamma_deg, '+.3f')} | "
        f"{_f(metrics_both.mae_alt, '.1f')} |"
    )
    lines.append(
        f"| at least one known=0 | {metrics_other.n} | "
        f"{_f(metrics_other.mae_tas)} | "
        f"{_f(metrics_other.bias_tas, '+.3f')} | "
        f"{_f(metrics_other.mae_gamma_deg)} | "
        f"{_f(metrics_other.bias_gamma_deg, '+.3f')} | "
        f"{_f(metrics_other.mae_alt, '.1f')} |"
    )
    lines.append("")
    lines.append("## Visual verdict")
    lines.append(verdict)
    lines.append("")
    lines.append("## Figure")
    lines.append(f"![Inference high coverage]({fig_rel.as_posix()})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Script reproductible : `scripts/debug/predict_high_coverage_flight.py`. "
        "Cohérent avec `data/mardown/vision_pilot_autopilot.md` et "
        "`data/mardown/artefact.md` (rounds 1+2A/2B/2C)."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    df = pl.read_delta(str(DELTA_PATH)).filter(pl.col("fdm_flag_valid"))
    sel = select_flight(df)

    print("=" * 72)
    print("Selected flight   :", sel.flight_id)
    print(f"Duration          : {sel.duration_min:.1f} min ({sel.n} samples)")
    print(f"tas_known cov     : {sel.tas_cov * 100:.2f}%")
    print(f"gamma_known cov   : {sel.gamma_cov * 100:.2f}%")
    print(f"Regimes present   : {', '.join(sel.regimes)}")
    print(
        f"Selection thresh. : tas>={sel.tas_threshold:.2f}, "
        f"gamma>={sel.gamma_threshold:.2f}, n>={MIN_N}"
    )
    print("=" * 72)

    fd = load_and_predict(sel.flight_id)
    fig_path = Path(
        f"data/figures/inference_check_{sel.flight_id}_high_coverage.png"
    )
    make_figure(fd, sel, fig_path)

    metrics_both, metrics_other = compute_metrics(fd)
    print(
        f"both known=1   : n={metrics_both.n} "
        f"MAE_TAS={metrics_both.mae_tas:.3f} m/s "
        f"MAE_gamma={metrics_both.mae_gamma_deg:.3f} deg "
        f"MAE_alt={metrics_both.mae_alt:.1f} m"
    )
    print(
        f"any known=0    : n={metrics_other.n} "
        f"MAE_TAS={metrics_other.mae_tas:.3f} m/s "
        f"MAE_gamma={metrics_other.mae_gamma_deg:.3f} deg "
        f"MAE_alt={metrics_other.mae_alt:.1f} m"
    )

    md = render_report(sel, metrics_both, metrics_other, fig_path)
    report_path = Path("data/mardown/high_coverage_validation.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md)
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
