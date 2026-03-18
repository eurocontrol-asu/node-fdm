"""Training command — ``fdm train``.

Ports logic from ``scripts/opensky/05_training.py`` into a typed CLI function.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import structlog

__all__ = ["run_training"]

log = structlog.get_logger()


def run_training(
    *,
    arch: str,
    config: Path,
    typecode: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    device: str = "cpu",
) -> None:
    """Train Neural ODE models for one or all typecodes.

    Args:
        arch: Architecture identifier (``"opensky"`` or ``"qar"``).
        config: Path to YAML pipeline config.
        typecode: Single typecode to train (default: all from config).
        epochs: Override number of training epochs.
        batch_size: Override batch size.
        lr: Override learning rate.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).
    """
    import polars as pl
    from node_fdm.loader import get_train_val_data
    from node_fdm.trainer import ODETrainer, TrainingConfig

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)

    # Trigger architecture auto-registration
    importlib.import_module(info.architecture_import)

    typecodes = [typecode] if typecode else cfg.typecodes
    process_dir = cfg.paths.resolve("process_dir")
    models_dir = cfg.paths.resolve("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    split_csv = process_dir / "dataset_split.csv"
    if not split_csv.exists():
        log.error("train_missing_split", path=str(split_csv))
        msg = f"dataset_split.csv not found at {split_csv}. Run 'fdm process' first."
        raise SystemExit(msg)

    split_df = pl.read_csv(split_csv)
    dx_col_names = [col for _, col in info.dx_cols]

    log.info(
        "train_start",
        arch=arch,
        typecodes=typecodes,
        device=device,
    )

    for acft in typecodes:
        log.info("train_typecode", typecode=acft)

        training_config = TrainingConfig(
            architecture_name=info.name,
            model_name=f"{info.name}_{acft}",
            model_params=(3, 2, 48),
            step=4.0,
            shift=60,
            lr=lr or 1e-3,
            weight_decay=1e-4,
            seq_len=60,
            batch_size=batch_size or 512,
            epochs=epochs or 800,
            method="euler",
            num_workers=4,
        )

        data_df = split_df.filter(pl.col("aircraft_type") == acft)

        if len(data_df) == 0:
            log.warning("train_empty_dataset", typecode=acft)
            continue

        train_ds, val_ds = get_train_val_data(
            data_df=data_df,
            x_cols=info.x_cols,
            u_cols=info.u_cols,
            e_cols=info.e0_cols,
            dx_cols=dx_col_names,
            seq_len=training_config.seq_len,
            shift=training_config.shift,
            preprocessing_fn=info.preprocessing_fn,
            segment_filter_fn=info.segment_filter_fn,
            train_limit=5000,
            val_limit=5000,
        )

        if epochs is None:
            n_step_per_epoch = max(len(train_ds) // training_config.batch_size, 1)
            coeff = min(50 / n_step_per_epoch, 10.0)
            adjusted_epochs = int(training_config.epochs * coeff)
            log.info(
                "train_epoch_adjust",
                typecode=acft,
                original_epochs=training_config.epochs,
                adjusted_epochs=adjusted_epochs,
                n_step_per_epoch=n_step_per_epoch,
                coeff=round(coeff, 3),
            )
            training_config = training_config.model_copy(
                update={"epochs": adjusted_epochs},
            )

        trainer = ODETrainer(
            config=training_config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=models_dir,
        )

        trainer.train()
        log.info("train_typecode_done", typecode=acft)

    log.info("train_done", typecodes=typecodes)
