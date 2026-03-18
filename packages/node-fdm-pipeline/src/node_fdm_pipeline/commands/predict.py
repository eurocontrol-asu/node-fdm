"""Prediction commands — ``fdm predict`` and ``fdm predict-bada``.

Ports logic from ``scripts/opensky/06_flight_prediction.py`` and
``scripts/opensky/07_bada_prediction.py`` into typed CLI functions.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import numpy as np

__all__ = ["run_predict", "run_predict_bada"]

log = structlog.get_logger()


def _filter_nan_segments(
    x_arr: np.ndarray,
    u_seq: np.ndarray,
    e_seq: np.ndarray,
    *,
    nan_threshold: float,
    flight_id: str,
    col_names: tuple[list[str], list[str], list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Filter arrays to finite-only rows, matching training NaN behavior.

    Args:
        x_arr: State array of shape ``(n_steps, n_x)``.
        u_seq: Control array of shape ``(n_steps, n_u)``.
        e_seq: Environment array of shape ``(n_steps, n_e)``.
        nan_threshold: Skip flight if NaN fraction exceeds this value.
        flight_id: Flight identifier for log messages.
        col_names: Tuple of ``(x_cols, u_cols, e_cols)`` for diagnostics.

    Returns:
        Tuple of ``(x_init, u_filtered, e_filtered)`` or *None* if the
        flight should be skipped (NaN fraction above threshold).
    """
    import numpy as np

    finite_mask = (
        np.isfinite(x_arr).all(axis=1)
        & np.isfinite(u_seq).all(axis=1)
        & np.isfinite(e_seq).all(axis=1)
    )
    nan_fraction = 1.0 - finite_mask.mean()

    if nan_fraction > nan_threshold:
        x_cols, u_cols, e_cols = col_names
        nan_cols = []
        for cols, arr in [(x_cols, x_arr), (u_cols, u_seq), (e_cols, e_seq)]:
            for i, col in enumerate(cols):
                if not np.isfinite(arr[:, i]).all():
                    nan_cols.append(col)
        log.warning(
            "predict_skip_nan",
            flight_id=flight_id,
            nan_pct=f"{nan_fraction:.1%}",
            nan_cols=nan_cols,
            threshold=f"{nan_threshold:.0%}",
        )
        return None

    return x_arr[finite_mask][0], u_seq[finite_mask], e_seq[finite_mask]


def run_predict(
    *,
    arch: str,
    config: Path,
    typecode: str | None = None,
    device: str = "cpu",
    local_model: bool = False,
    nan_threshold: float = 0.8,
) -> None:
    """Predict flight trajectories using trained Neural ODE models.

    Args:
        arch: Architecture identifier (``"opensky"`` or ``"qar"``).
        config: Path to YAML pipeline config.
        typecode: Single typecode to predict (default: all from config).
        device: PyTorch device string.
        local_model: Use local model directory instead of packaged pretrained.
        nan_threshold: Maximum fraction of NaN rows before skipping a flight.
            Flights where NaN fraction exceeds this value are skipped entirely.
            Default ``0.8`` (skip if >80% of timesteps contain NaN).
    """
    import numpy as np
    import polars as pl
    from node_fdm.predictor import NodeFDMPredictor

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)

    typecodes = [typecode] if typecode else cfg.typecodes
    process_dir = cfg.paths.resolve("process_dir")
    models_dir = cfg.paths.resolve("models_dir")
    predict_dir = cfg.paths.resolve("predicted_dir")
    predict_dir.mkdir(parents=True, exist_ok=True)

    split_csv = process_dir / "dataset_split.csv"
    if not split_csv.exists():
        log.error("predict_missing_split", path=str(split_csv))
        msg = f"dataset_split.csv not found at {split_csv}. Run 'fdm process' first."
        raise SystemExit(msg)

    split_df = pl.read_csv(split_csv)

    log.info(
        "predict_start",
        arch=arch,
        typecodes=typecodes,
        device=device,
        local_model=local_model,
    )

    for acft in typecodes:
        log.info("predict_typecode", typecode=acft)

        if local_model:
            model_path = models_dir / f"{info.name}_{acft}"
        else:
            model_path = Path(
                str(
                    files(f"node_fdm.models.pretrained_models.{info.name}").joinpath(
                        f"{info.name}_{acft}"
                    )
                )
            )

        if not model_path.exists():
            log.warning("predict_model_not_found", typecode=acft, path=str(model_path))
            continue

        predictor = NodeFDMPredictor(model_path=model_path, device=device)

        data_df = split_df.filter(pl.col("aircraft_type") == acft)
        test_df = data_df.filter(pl.col("split") == "test")

        if len(test_df) == 0:
            log.warning("predict_empty_test_set", typecode=acft)
            continue

        output_dir = predict_dir / acft
        output_dir.mkdir(parents=True, exist_ok=True)

        for row in test_df.iter_rows(named=True):
            flight_path = Path(row["filepath"])
            flight_id = flight_path.stem

            raw = pl.read_parquet(flight_path)
            processed = info.preprocessing_fn(raw)

            # Extract arrays for predictor (float32 numpy)
            arrays = {
                "x": processed.select(info.x_cols).to_numpy().astype(np.float32),
                "u": processed.select(info.u_cols).to_numpy().astype(np.float32),
                "e": processed.select(info.e0_cols).to_numpy().astype(np.float32),
            }

            # --- NaN segment filter (match training behavior) ---
            result = _filter_nan_segments(
                arrays["x"],
                arrays["u"],
                arrays["e"],
                nan_threshold=nan_threshold,
                flight_id=flight_id,
                col_names=(info.x_cols, info.u_cols, info.e0_cols),
            )
            if result is None:
                continue

            predictions = predictor.predict_flight(*result)

            pred_df = pl.DataFrame({f"pred_{k}": v for k, v in predictions.items()})
            pred_df.write_parquet(output_dir / f"{flight_id}.parquet")

        log.info("predict_typecode_done", typecode=acft)

    log.info("predict_done", typecodes=typecodes)


def run_predict_bada(
    *,
    config: Path,
    typecode: str | None = None,
    jobs: int | None = None,
) -> None:
    """Run BADA 4.2 baseline predictions.

    Args:
        config: Path to YAML pipeline config.
        typecode: Single typecode to predict (default: all from config).
        jobs: Number of parallel workers (default: from config computing section).
    """
    import polars as pl
    from node_fdm_bada.aircraft_mapping import get_bada_identifier
    from node_fdm_bada.predictor import process_single_flight
    from node_fdm_data.preprocessing.opensky import flight_processing
    from node_fdm_data.processor import FlightProcessor

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)

    typecodes = [typecode] if typecode else cfg.typecodes
    process_dir = cfg.paths.resolve("process_dir")
    bada_dir = cfg.paths.resolve("bada_dir")
    bada_dir.mkdir(parents=True, exist_ok=True)
    bada_4_2_dir = cfg.bada.bada_4_2_dir

    split_csv = process_dir / "dataset_split.csv"
    if not split_csv.exists():
        log.error("predict_bada_missing_split", path=str(split_csv))
        msg = f"dataset_split.csv not found at {split_csv}. Run 'fdm process' first."
        raise SystemExit(msg)

    split_df = pl.read_csv(split_csv)
    processor = FlightProcessor(steps=[flight_processing])
    n_jobs = jobs or cfg.computing.default_cpu_count

    log.info("predict_bada_start", typecodes=typecodes, jobs=n_jobs)

    for acft in typecodes:
        log.info("predict_bada_typecode", typecode=acft)
        try:
            bada_name = get_bada_identifier(acft)
        except KeyError:
            log.warning("predict_bada_no_mapping", typecode=acft)
            continue

        try:
            from pyBADA.bada4 import Bada4Aircraft

            ac = Bada4Aircraft("4.2", filePath=str(bada_4_2_dir), acName=bada_name)
        except Exception:  # noqa: BLE001
            log.warning("predict_bada_load_failed", typecode=acft, bada_name=bada_name)
            continue

        data_df = split_df.filter(pl.col("aircraft_type") == acft)
        test_df = data_df.filter(pl.col("split") == "test")

        if len(test_df) == 0:
            log.warning("predict_bada_empty_test_set", typecode=acft)
            continue

        output_dir = bada_dir / acft
        output_dir.mkdir(parents=True, exist_ok=True)

        filepaths = test_df["filepath"].to_list()

        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_single_flight)(fp, ac, processor, output_dir) for fp in filepaths
        )

        log.info("predict_bada_typecode_done", typecode=acft)

    log.info("predict_bada_done", typecodes=typecodes)
