"""Evaluation command — ``fdm evaluate``.

Ports logic from ``scripts/opensky/09_performance_aggregation.py`` into a
typed CLI function with a reusable ``compute_errors_by_phase`` utility.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

__all__ = ["compute_errors_by_phase", "run_evaluate"]

log = structlog.get_logger()

# Minimum altitude (ft) to filter ground-level data from evaluation
_MIN_ALTITUDE_FT = 5000


def compute_errors_by_phase(
    df: pl.DataFrame,
    pred_col: str,
    target_col: str,
    vertical_rate_col: str = "vz_ms",
    eps: float = 1e-8,
) -> pl.DataFrame:
    """Compute MAE, MAPE, ME and their std by flight phase.

    Phases are determined by vertical rate: climb (>1 m/s), descent (<-1 m/s),
    and level flight (in between).

    Args:
        df: DataFrame with prediction and ground-truth columns.
        pred_col: Name of the prediction column.
        target_col: Name of the ground-truth column.
        vertical_rate_col: Column with vertical rate in m/s.
        eps: Small value to avoid division by zero in MAPE.

    Returns:
        DataFrame with Phase, MAE, MAE_std, MAPE, MAPE_std, ME, ME_std, Count.
    """
    vz = df[vertical_rate_col].to_numpy()
    climb_mask = vz > 1.0
    descent_mask = vz < -1.0
    level_mask = (~climb_mask) & (~descent_mask)

    phases = {
        "All phases": np.ones(len(df), dtype=bool),
        "Climb": climb_mask,
        "Level flight": level_mask,
        "Descent": descent_mask,
    }

    is_angle = "gamma" in pred_col.lower()
    results: list[tuple[str, float, float, float, float, float, float, int]] = []

    for phase, mask in phases.items():
        y_pred = df.filter(pl.Series(mask))[pred_col].to_numpy()
        y_true = df.filter(pl.Series(mask))[target_col].to_numpy()

        valid = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[valid], y_true[valid]
        if len(y_true) == 0:
            continue

        if is_angle:
            deg_factor = 180 / np.pi
            y_pred = y_pred * deg_factor
            y_true = y_true * deg_factor

        err = y_pred - y_true
        abs_err = np.abs(err)

        if is_angle:
            abs_perc_err = np.full_like(abs_err, np.nan)
        else:
            abs_perc_err = np.abs(err / (y_true + eps)) * 100.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mape_mean = float(np.nanmean(abs_perc_err))
            mape_std = float(np.nanstd(abs_perc_err))

        results.append(
            (
                phase,
                float(np.mean(abs_err)),
                float(np.std(abs_err)),
                mape_mean,
                mape_std,
                float(np.mean(err)),
                float(np.std(err)),
                len(y_true),
            )
        )

    return pl.DataFrame(
        results,
        schema=[
            ("Phase", pl.Utf8),
            ("MAE", pl.Float64),
            ("MAE_std", pl.Float64),
            ("MAPE (%)", pl.Float64),
            ("MAPE_std", pl.Float64),
            ("ME", pl.Float64),
            ("ME_std", pl.Float64),
            ("Count", pl.Int64),
        ],
        orient="row",
    )


def _evaluate_typecode(
    acft: str,
    *,
    process_dir: Path,
    predict_dir: Path,
    bada_dir: Path,
    processor: Any,
    variables: dict[str, str],
) -> list[pl.DataFrame]:
    """Collect error metrics for a single typecode."""
    acft_bada = bada_dir / acft
    acft_pred = predict_dir / acft
    if not acft_bada.exists() and not acft_pred.exists():
        log.info("evaluate_skip_typecode", typecode=acft, reason="no predictions")
        return []

    source_dir = acft_bada if acft_bada.exists() else acft_pred
    parquet_files = sorted(source_dir.glob("*.parquet"))
    if not parquet_files:
        return []

    log.info("evaluate_typecode", typecode=acft, flights=len(parquet_files))

    acft_frames: list[pl.DataFrame] = []
    for file in parquet_files:
        try:
            gt_path = process_dir / acft / file.name
            if not gt_path.exists():
                continue
            f = pl.read_parquet(gt_path)
            processor.process(f).collect()

            pred_path = predict_dir / acft / file.name
            if pred_path.exists():
                f = f.hstack(pl.read_parquet(pred_path))

            bada_path = bada_dir / acft / file.name
            if bada_path.exists():
                f = f.hstack(pl.read_parquet(bada_path))

            if "altitude" in f.columns:
                f = f.filter(pl.col("altitude") > _MIN_ALTITUDE_FT)

            acft_frames.append(f)
        except Exception:  # noqa: BLE001
            log.debug("evaluate_file_error", file=str(file))
            continue

    if not acft_frames:
        return []

    df_acft = pl.concat(acft_frames, how="vertical_relaxed")
    results: list[pl.DataFrame] = []

    for var, label in variables.items():
        for prefix in ["bada_", "pred_"]:
            pred_col = f"{prefix}{var}"
            if pred_col not in df_acft.columns or var not in df_acft.columns:
                continue
            if "vz_ms" not in df_acft.columns:
                continue
            metrics = compute_errors_by_phase(df_acft, pred_col=pred_col, target_col=var)
            metrics = metrics.with_columns(
                pl.lit(acft).alias("Aircraft"),
                pl.lit(label).alias("Variable"),
                pl.lit(prefix[:-1].upper()).alias("Model"),
            )
            results.append(metrics)

    return results


def run_evaluate(
    *,
    arch: str,
    config: Path,
) -> None:
    """Compute prediction error metrics per flight phase.

    Loads ground-truth, Node-FDM predictions, and BADA predictions,
    then computes MAE/MAPE/ME by phase for each variable and model.

    Args:
        arch: Architecture identifier (``"opensky"`` or ``"qar"``).
        config: Path to YAML pipeline config.
    """
    from node_fdm_data.preprocessing.opensky import flight_processing
    from node_fdm_data.processor import FlightProcessor

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    _info = resolve_architecture(arch)

    process_dir = cfg.paths.resolve("process_dir")
    predict_dir = cfg.paths.resolve("predicted_dir")
    bada_dir = cfg.paths.resolve("bada_dir")

    processor = FlightProcessor(steps=[flight_processing])

    variables = {
        "alt_std_m": "Altitude [m]",
        "tas_ms": "True airspeed [m/s]",
        "gamma_rad": "Flight path angle [deg]",
    }

    all_results: list[pl.DataFrame] = []

    log.info("evaluate_start", arch=arch, typecodes=cfg.typecodes)

    for acft in cfg.typecodes:
        results = _evaluate_typecode(
            acft,
            process_dir=process_dir,
            predict_dir=predict_dir,
            bada_dir=bada_dir,
            processor=processor,
            variables=variables,
        )
        all_results.extend(results)

    if not all_results:
        log.warning("evaluate_no_results", msg="No predictions found to evaluate")
        return

    final_df = pl.concat(all_results, how="vertical")
    final_df = final_df.select(
        "Aircraft",
        "Variable",
        "Phase",
        "Model",
        "MAE",
        "MAE_std",
        "MAPE (%)",
        "MAPE_std",
        "ME",
        "ME_std",
        "Count",
    )

    phase_order = {"All phases": 0, "Climb": 1, "Level flight": 2, "Descent": 3}
    final_df = (
        final_df.with_columns(
            pl.col("Phase").replace_strict(phase_order, default=99).alias("_phase_order")
        )
        .sort("Aircraft", "Variable", "_phase_order", "Model")
        .drop("_phase_order")
    )

    output = cfg.paths.data_dir / "performance.parquet"
    final_df.write_parquet(output)
    log.info("evaluate_done", rows=len(final_df), output=str(output))
