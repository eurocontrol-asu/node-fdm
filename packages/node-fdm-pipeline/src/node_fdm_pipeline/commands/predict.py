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


def _resolve_model_path(
    *,
    info: object,
    acft: str,
    local_model: bool,
    models_dir: Path,
    model_name: str | None = None,
) -> Path:
    """Resolve the model directory.

    Precedence:
        * ``local_model=True`` and ``model_name`` provided →
          ``models_dir / model_name``.
        * ``local_model=True`` and ``model_name`` is ``None`` →
          ``models_dir / f"{info.name}_{acft}"`` (legacy default).
        * ``local_model=False`` → packaged pretrained directory under
          ``node_fdm.models.pretrained_models.<info.name>``.

    Args:
        info: Architecture info (must expose ``.name``).
        acft: ICAO typecode (used in the legacy default).
        local_model: If True, resolve under ``models_dir``; else packaged.
        models_dir: Local models root.
        model_name: Optional explicit checkpoint directory name (relative
            to ``models_dir``).  Only honoured when ``local_model=True``.
    """
    if local_model:
        if model_name is not None:
            return models_dir / model_name
        return models_dir / f"{info.name}_{acft}"  # type: ignore[attr-defined]
    return Path(
        str(
            files(f"node_fdm.models.pretrained_models.{info.name}").joinpath(  # type: ignore[attr-defined]
                f"{info.name}_{acft}"  # type: ignore[attr-defined]
            )
        )
    )


def _load_test_df(delta_table: Path) -> object:
    """Read the Delta table and keep valid test-split rows, with sel_* columns NaN/null filled."""
    import polars as pl
    from node_fdm_data.delta import read_delta_table

    df = read_delta_table(delta_table)
    df = df.filter(pl.col("fdm_flag_valid") & pl.col("meta_split").eq("test"))
    sel_cols = [c for c in df.columns if c.startswith("fdm_") and "_sel_" in c]
    if sel_cols:
        df = df.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols])
    return df


def _predict_flight(
    *,
    flight_df: object,
    info: object,
    predictor: object,
    output_dir: Path,
    nan_threshold: float,
) -> None:
    """Run the predictor on one flight and write its predictions parquet to output_dir."""
    import numpy as np
    import polars as pl

    flight_id = flight_df["meta_flight_id"][0]  # type: ignore[index]
    x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)  # type: ignore[attr-defined]
    u_arr = flight_df.select(info.u_cols).to_numpy().astype(np.float32)  # type: ignore[attr-defined]
    e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)  # type: ignore[attr-defined]

    result = _filter_nan_segments(
        x_arr,
        u_arr,
        e_arr,
        nan_threshold=nan_threshold,
        flight_id=flight_id,
        col_names=(info.x_cols, info.u_cols, info.e0_cols),  # type: ignore[attr-defined]
    )
    if result is None:
        return

    try:
        predictions = predictor.predict_flight(*result)  # type: ignore[attr-defined]
    except ValueError:
        log.warning("predict_skip_bad_x_init", flight_id=flight_id)
        return

    pred_df = pl.DataFrame({f"pred_{k}": v for k, v in predictions.items()})
    pred_df.write_parquet(output_dir / f"{flight_id}.parquet")


def _predict_typecode(
    *,
    acft: str,
    df: object,
    info: object,
    models_dir: Path,
    predict_dir: Path,
    device: str,
    local_model: bool,
    nan_threshold: float,
    model_name: str | None = None,
) -> None:
    """Load the typecode's model and predict every flight in its filtered test partition."""
    import polars as pl
    from node_fdm.predictor import NodeFDMPredictor

    log.info("predict_typecode", typecode=acft)
    model_path = _resolve_model_path(
        info=info,
        acft=acft,
        local_model=local_model,
        models_dir=models_dir,
        model_name=model_name,
    )
    if not model_path.exists():
        log.warning("predict_model_not_found", typecode=acft, path=str(model_path))
        return

    predictor = NodeFDMPredictor(model_path=model_path, device=device)
    acft_df = df.filter(pl.col("meta_aircraft_type") == acft)  # type: ignore[attr-defined]
    if len(acft_df) == 0:
        log.warning("predict_empty_test_set", typecode=acft)
        return

    output_dir = predict_dir / acft
    output_dir.mkdir(parents=True, exist_ok=True)

    for flight_df in acft_df.partition_by("meta_flight_id", maintain_order=True):
        _predict_flight(
            flight_df=flight_df,
            info=info,
            predictor=predictor,
            output_dir=output_dir,
            nan_threshold=nan_threshold,
        )

    log.info("predict_typecode_done", typecode=acft)


def run_predict(
    *,
    arch: str,
    config: Path,
    typecode: str | None = None,
    device: str = "cpu",
    local_model: bool = False,
    nan_threshold: float = 0.8,
    model_name: str | None = None,
) -> None:
    """Predict flight trajectories using trained Neural ODE models.

    Reads from the Delta Table (v3 pipeline) — data is already in SI
    units with ``meta_split`` and ``fdm_flag_valid`` columns.

    Args:
        arch: Architecture identifier (``"qar"`` or ``"adsb"``).
        config: Path to YAML pipeline config.
        typecode: Single typecode to predict (default: all from config).
        device: PyTorch device string.
        local_model: Use local model directory instead of packaged pretrained.
        nan_threshold: Maximum fraction of NaN rows before skipping a flight.
            Flights where NaN fraction exceeds this value are skipped entirely.
            Default ``0.8`` (skip if >80% of timesteps contain NaN).
        model_name: Optional explicit checkpoint directory name relative to
            ``models_dir``.  Only used when ``local_model`` is ``True``;
            overrides the default ``f"{info.name}_{typecode}"`` convention,
            so checkpoints written by ``fdm train --model-name <name>`` can
            be loaded back without renaming.
    """
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)

    typecodes = [typecode] if typecode else cfg.typecodes
    models_dir = cfg.paths.resolve("models_dir")
    predict_dir = cfg.paths.resolve("predicted_dir")
    predict_dir.mkdir(parents=True, exist_ok=True)
    delta_table = cfg.paths.resolve("delta_table")

    df = _load_test_df(delta_table)

    log.info(
        "predict_start",
        arch=arch,
        typecodes=typecodes,
        device=device,
        local_model=local_model,
        rows=len(df),  # type: ignore[arg-type]
    )

    for acft in typecodes:
        _predict_typecode(
            acft=acft,
            df=df,
            info=info,
            models_dir=models_dir,
            predict_dir=predict_dir,
            device=device,
            local_model=local_model,
            nan_threshold=nan_threshold,
            model_name=model_name,
        )

    log.info("predict_done", typecodes=typecodes)


def run_predict_bada(
    *,
    config: Path,
    typecode: str | None = None,
    jobs: int | None = None,
) -> None:
    """Run BADA 4.2 baseline predictions.

    Reads from the Delta Table (v3 pipeline) — data is already processed.
    Per-flight parquets are written to a temporary directory for the BADA
    predictor interface (which expects file paths).

    Args:
        config: Path to YAML pipeline config.
        typecode: Single typecode to predict (default: all from config).
        jobs: Number of parallel workers (default: from config computing section).
    """
    import tempfile

    import polars as pl
    from node_fdm_bada.aircraft_mapping import get_bada_identifier
    from node_fdm_bada.predictor import process_single_flight
    from node_fdm_data.delta import read_delta_table
    from node_fdm_data.preprocessing.opensky import flight_processing
    from node_fdm_data.processor import FlightProcessor

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)

    typecodes = [typecode] if typecode else cfg.typecodes
    delta_table = cfg.paths.resolve("delta_table")
    bada_dir = cfg.paths.resolve("bada_dir")
    bada_dir.mkdir(parents=True, exist_ok=True)
    bada_4_2_dir = cfg.bada.bada_4_2_dir

    # Read Delta Table — filter on valid + test split
    df = read_delta_table(delta_table)
    df = df.filter(
        pl.col("fdm_flag_valid") & pl.col("meta_split").eq("test"),
    )

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
            from pyBADA.bada4 import (
                Bada4Aircraft,  # type: ignore[import-not-found,import-untyped,unused-ignore]
            )

            ac = Bada4Aircraft("4.2", filePath=str(bada_4_2_dir), acName=bada_name)
        except Exception:  # noqa: BLE001
            log.warning("predict_bada_load_failed", typecode=acft, bada_name=bada_name)
            continue

        acft_df = df.filter(pl.col("meta_aircraft_type") == acft)
        if len(acft_df) == 0:
            log.warning("predict_bada_empty_test_set", typecode=acft)
            continue

        output_dir = bada_dir / acft
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write per-flight parquets for BADA predictor interface
        flights = acft_df.partition_by("meta_flight_id", maintain_order=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepaths: list[str] = []
            for flight_df in flights:
                fid = flight_df["meta_flight_id"][0]
                fp = Path(tmp_dir) / f"{fid}.parquet"
                flight_df.write_parquet(fp)
                filepaths.append(str(fp))

            from joblib import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
                Parallel,
                delayed,
            )

            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_single_flight)(fp, ac, processor, output_dir) for fp in filepaths
            )

        log.info("predict_bada_typecode_done", typecode=acft)

    log.info("predict_bada_done", typecodes=typecodes)
