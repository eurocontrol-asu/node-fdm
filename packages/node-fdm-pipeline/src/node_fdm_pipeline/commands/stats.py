"""Dataset statistics command — ``fdm dataset-stats``.

Ports logic from ``scripts/opensky/10_dataset_stats.py`` into a typed CLI function.
"""

from __future__ import annotations

from pathlib import Path

import structlog

__all__ = ["run_dataset_stats"]

log = structlog.get_logger()

# Constants for computing flight hours from segment counts
_SEGMENT_DURATION_S = 4  # seconds per segment (sample rate)
_SEQ_LEN = 60  # sequence length used in dataset creation


def run_dataset_stats(
    *,
    arch: str,
    config: Path,
) -> None:
    """Compute dataset statistics: flight counts and hours per split.

    Args:
        arch: Architecture identifier (``"opensky"`` or ``"qar"``).
        config: Path to YAML pipeline config.
    """
    import polars as pl
    from node_fdm.loader import get_train_val_data

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)

    delta_table = cfg.paths.resolve("delta_table")
    if not delta_table.exists():
        log.error("stats_missing_delta", path=str(delta_table))
        msg = f"Delta table not found at {delta_table}. Run the pipeline first."
        raise SystemExit(msg)

    full_df = pl.read_delta(str(delta_table))
    full_df = full_df.filter(pl.col("fdm_flag_valid"))

    dx_col_names = [col for _, col in info.dx_cols]

    log.info("stats_start", arch=arch, typecodes=cfg.typecodes)

    for acft in cfg.typecodes:
        data_df = full_df.filter(pl.col("meta_aircraft_type") == acft)

        train_flights = (
            data_df.filter(pl.col("meta_split") == "train").get_column("meta_flight_id").n_unique()
        )
        val_flights = (
            data_df.filter(pl.col("meta_split") == "val").get_column("meta_flight_id").n_unique()
        )
        test_flights = (
            data_df.filter(pl.col("meta_split") == "test").get_column("meta_flight_id").n_unique()
        )

        train_ds, val_ds = get_train_val_data(
            data_df=data_df,
            x_cols=info.x_cols,
            u_cols=info.u_cols,
            e_cols=info.e0_cols,
            dx_cols=dx_col_names,
            seq_len=_SEQ_LEN,
            shift=_SEQ_LEN,
            train_limit=None,
            val_limit=None,
        )

        train_hours = round(len(train_ds) * _SEQ_LEN * _SEGMENT_DURATION_S / 3600, 2)
        val_hours = round(len(val_ds) * _SEQ_LEN * _SEGMENT_DURATION_S / 3600, 2)

        log.info(
            "stats_typecode",
            typecode=acft,
            train_flights=train_flights,
            train_hours=train_hours,
            val_flights=val_flights,
            val_hours=val_hours,
            test_flights=test_flights,
        )

    log.info("stats_done", typecodes=cfg.typecodes)
