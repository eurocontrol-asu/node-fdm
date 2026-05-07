"""Evaluation command — ``fdm evaluate``.

Ports logic from ``scripts/opensky/09_performance_aggregation.py`` into a
typed CLI function with a reusable ``compute_errors_by_phase`` utility.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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

    is_heading = "heading" in pred_col.lower()
    is_gamma = "gamma" in pred_col.lower()
    is_angle = is_heading or is_gamma
    results: list[tuple[str, float, float, float, float, float, float, int]] = []

    for phase, mask in phases.items():
        y_pred = df.filter(pl.Series(mask))[pred_col].to_numpy()
        y_true = df.filter(pl.Series(mask))[target_col].to_numpy()

        valid = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[valid], y_true[valid]
        if len(y_true) == 0:
            continue

        deg_factor = 180 / np.pi
        if is_heading:
            err_rad = (y_pred - y_true + np.pi) % (2 * np.pi) - np.pi
            err = err_rad * deg_factor
        elif is_gamma:
            y_pred = y_pred * deg_factor
            y_true = y_true * deg_factor
            err = y_pred - y_true
        else:
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


def _load_flight_frame(
    flight_df: pl.DataFrame,
    pred_path: Path,
    bada_path: Path,
) -> pl.DataFrame | None:
    """Hstack node-fdm and BADA predictions onto a flight, filtered above the min altitude."""
    if not pred_path.exists() and not bada_path.exists():
        return None
    f = flight_df
    if pred_path.exists():
        f = f.hstack(pl.read_parquet(pred_path))
    if bada_path.exists():
        f = f.hstack(pl.read_parquet(bada_path))
    if "raw_alt_ft" in f.columns:
        f = f.filter(pl.col("raw_alt_ft") > _MIN_ALTITUDE_FT)
    return f


def _collect_acft_frames(
    acft: str,
    flights: list[pl.DataFrame],
    predict_acft_dir: Path,
    bada_dir: Path,
) -> list[pl.DataFrame]:
    """Build per-flight evaluation frames for a typecode, skipping flights that error out.

    When both prediction sources exist, restrict to flights predicted by
    both (intersection) so NODE and BADA are evaluated on the same set.
    """
    bada_acft_dir = bada_dir / acft
    require_both = predict_acft_dir.exists() and bada_acft_dir.exists()
    acft_frames: list[pl.DataFrame] = []
    skipped = 0
    for flight_df in flights:
        fid = flight_df["meta_flight_id"][0]
        try:
            fname = f"{fid}.parquet"
            pred_path = predict_acft_dir / fname
            bada_path = bada_acft_dir / fname
            if require_both and not (pred_path.exists() and bada_path.exists()):
                skipped += 1
                continue
            f = _load_flight_frame(flight_df, pred_path, bada_path)
            if f is not None:
                acft_frames.append(f)
        except Exception:  # noqa: BLE001
            log.debug("evaluate_flight_error", flight_id=fid)
    if require_both and skipped:
        log.info("evaluate_intersection", typecode=acft, skipped=skipped, kept=len(acft_frames))
    return acft_frames


def _metrics_for_variable(
    df_acft: pl.DataFrame,
    acft: str,
    var: str,
    label: str,
    prefix: str,
) -> pl.DataFrame | None:
    """Compute per-phase error metrics for one (variable, model) pair.

    Tagged with Aircraft/Variable/Model.
    """
    pred_col = f"{prefix}{var}"
    if pred_col not in df_acft.columns or var not in df_acft.columns:
        return None
    if "raw_vz_ms" not in df_acft.columns:
        return None
    metrics = compute_errors_by_phase(
        df_acft,
        pred_col=pred_col,
        target_col=var,
        vertical_rate_col="raw_vz_ms",
    )
    return metrics.with_columns(
        pl.lit(acft).alias("Aircraft"),
        pl.lit(label).alias("Variable"),
        pl.lit(prefix[:-1].upper()).alias("Model"),
    )


def _position_metrics_for_model(
    df_acft: pl.DataFrame,
    acft: str,
    prefix: str,
) -> pl.DataFrame | None:
    """Compute per-phase haversine position errors for one model.

    Tagged with Aircraft/Variable="Position [m]"/Model.
    """
    lat_pred_col = f"{prefix}lat_deg"
    lon_pred_col = f"{prefix}lon_deg"
    metrics = compute_position_errors_by_phase(
        df_acft,
        lat_pred_col=lat_pred_col,
        lon_pred_col=lon_pred_col,
        lat_true_col="lat_deg",
        lon_true_col="lon_deg",
        vertical_rate_col="raw_vz_ms",
    )
    if metrics is None or len(metrics) == 0:
        return None
    return metrics.with_columns(
        pl.lit(acft).alias("Aircraft"),
        pl.lit("Position [m]").alias("Variable"),
        pl.lit(prefix[:-1].upper()).alias("Model"),
    )


def compute_position_errors_by_phase(
    df: pl.DataFrame,
    lat_pred_col: str,
    lon_pred_col: str,
    lat_true_col: str,
    lon_true_col: str,
    vertical_rate_col: str = "vz_ms",
) -> pl.DataFrame | None:
    """Compute haversine position error metrics by flight phase.

    Returns the same schema as ``compute_errors_by_phase``: per-phase MAE,
    MAE_std, MAPE (NaN for distances), MAPE_std, ME, ME_std, Count. Returns
    ``None`` if any of the required lat/lon/vz columns are missing.
    """
    required = (lat_pred_col, lon_pred_col, lat_true_col, lon_true_col, vertical_rate_col)
    if any(col not in df.columns for col in required):
        return None

    earth_radius_m = 6_371_000.0
    lat1 = np.radians(df[lat_pred_col].to_numpy())
    lon1 = np.radians(df[lon_pred_col].to_numpy())
    lat2 = np.radians(df[lat_true_col].to_numpy())
    lon2 = np.radians(df[lon_true_col].to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distance = 2.0 * earth_radius_m * np.arcsin(np.minimum(1.0, np.sqrt(a)))

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

    results: list[tuple[str, float, float, float, float, float, float, int]] = []
    for phase, mask in phases.items():
        d = distance[mask]
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        abs_perc_err = np.full_like(d, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mape_mean = float(np.nanmean(abs_perc_err))
            mape_std = float(np.nanstd(abs_perc_err))
        results.append(
            (
                phase,
                float(np.mean(d)),
                float(np.std(d)),
                mape_mean,
                mape_std,
                float(np.mean(d)),
                float(np.std(d)),
                len(d),
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


def evaluate_typecode(
    acft: str,
    *,
    acft_df: pl.DataFrame,
    predict_acft_dir: Path,
    bada_dir: Path,
    variables: dict[str, str],
) -> list[pl.DataFrame]:
    """Collect error metrics for a single typecode.

    Ground truth comes from the Delta Table (``acft_df``).  Prediction
    and BADA output files are matched by ``meta_flight_id``.
    """
    if not (bada_dir / acft).exists() and not predict_acft_dir.exists():
        log.info("evaluate_skip_typecode", typecode=acft, reason="no predictions")
        return []

    flights = acft_df.partition_by("meta_flight_id", maintain_order=True)
    log.info("evaluate_typecode", typecode=acft, flights=len(flights))

    acft_frames = _collect_acft_frames(acft, flights, predict_acft_dir, bada_dir)
    if not acft_frames:
        return []

    df_acft = pl.concat(acft_frames, how="vertical_relaxed")
    results: list[pl.DataFrame] = []
    for var, label in variables.items():
        for prefix in ("bada_", "pred_"):
            metrics = _metrics_for_variable(df_acft, acft, var, label, prefix)
            if metrics is not None:
                results.append(metrics)
    for prefix in ("bada_", "pred_"):
        position_metrics = _position_metrics_for_model(df_acft, acft, prefix)
        if position_metrics is not None:
            results.append(position_metrics)
    return results


def run_evaluate(
    *,
    arch: str,
    config: Path,
    model_name: str | None = None,
) -> None:
    """Compute prediction error metrics per flight phase.

    Reads ground truth from the Delta Table (v3 pipeline), then loads
    per-flight predictions and BADA outputs for comparison.

    Args:
        arch: Architecture identifier (``"qar"`` or ``"adsb"``).
        config: Path to YAML pipeline config.
    """
    from node_fdm_data.delta import read_delta_table

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)

    delta_table = cfg.paths.resolve("delta_table")
    predict_dir = cfg.paths.resolve("predicted_dir")
    bada_dir = cfg.paths.resolve("bada_dir")

    # Read Delta Table — filter on valid + test split
    df = read_delta_table(delta_table)
    df = df.filter(
        pl.col("fdm_flag_valid") & pl.col("meta_split").eq("test"),
    )

    variables = {
        "raw_alt_m": "Altitude [m]",
        "era_tas_ms": "True airspeed [m/s]",
        "fdm_gamma_rad": "Flight path angle [deg]",
        "fdm_heading_rad": "Heading [deg]",
    }

    all_results: list[pl.DataFrame] = []

    log.info("evaluate_start", arch=arch, typecodes=cfg.typecodes)

    for acft in cfg.typecodes:
        acft_df = df.filter(pl.col("meta_aircraft_type") == acft)
        sub = model_name if model_name is not None else f"{info.name}_{acft}"
        results = evaluate_typecode(
            acft,
            acft_df=acft_df,
            predict_acft_dir=predict_dir / sub / acft,
            bada_dir=bada_dir,
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

    out_sub = model_name if model_name is not None else info.name
    output_dir = cfg.paths.data_dir / "model_performance" / out_sub
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "performance.parquet"
    final_df.write_parquet(output)
    log.info("evaluate_done", rows=len(final_df), output=str(output))

    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        float_precision=3,
    ):
        print(final_df)  # noqa: T201
