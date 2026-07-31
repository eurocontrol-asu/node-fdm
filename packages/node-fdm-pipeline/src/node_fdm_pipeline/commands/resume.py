"""Resume training from an existing checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from node_fdm_pipeline.resolver import resolve_architecture

if TYPE_CHECKING:
    import polars as pl

__all__ = ["run_resume"]

log = structlog.get_logger("node_fdm_pipeline.commands.resume")


@dataclass(frozen=True)
class _Overrides:
    """CLI override values applied on top of the saved meta.json training config."""

    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    seq_len: int | None = None
    shift: int | None = None
    method: str | None = None
    model_name: str | None = None
    lambda_tracking: float | None = None
    use_mode_weights: bool | None = None


def _load_meta(model: Path) -> Any:
    """Load ModelMeta from the checkpoint's meta.json or exit with an error."""
    from node_fdm.predictor import ModelMeta

    meta_path = model / "meta.json"
    if not meta_path.exists():
        log.error("resume_missing_meta", path=str(meta_path))
        msg = f"meta.json not found at {meta_path}"
        raise SystemExit(msg)
    return ModelMeta.from_json(meta_path)


def _resolve_arch_info(architecture_name: str) -> Any:
    """Resolve a checkpoint architecture through installed providers."""
    try:
        return resolve_architecture(architecture_name)
    except ValueError as exc:
        msg = f"Unknown architecture_name in meta.json: {architecture_name!r}."
        raise SystemExit(msg) from exc


def _load_data(delta_table: Path, typecode_suffix: str) -> pl.DataFrame:
    """Read the Delta table, filter to valid rows of the given typecode, or exit if empty."""
    import polars as pl

    if not delta_table.exists():
        log.error("resume_missing_delta", path=str(delta_table))
        msg = f"Delta table not found at {delta_table}. Run the pipeline first."
        raise SystemExit(msg)

    full_df = pl.read_delta(str(delta_table))
    full_df = full_df.filter(pl.col("fdm_flag_valid"))
    data_df = full_df.filter(pl.col("meta_aircraft_type") == typecode_suffix)

    if len(data_df) == 0:
        log.warning("resume_empty_dataset", typecode=typecode_suffix)
        msg = f"No data for typecode {typecode_suffix!r}."
        raise SystemExit(msg)
    return data_df


def _build_training_config(
    meta: Any,
    model_name: str,
    ov: _Overrides,
    *,
    cfg_use_mode_weights: bool = False,
    cfg_mode_weight_alpha: float = 0.5,
) -> Any:
    """Build a TrainingConfig from saved meta values overlaid with CLI overrides."""
    from node_fdm.trainer import TrainingConfig

    effective_seq_len = ov.seq_len or meta.seq_len
    effective_shift = ov.shift or effective_seq_len
    use_mode_weights = (
        ov.use_mode_weights if ov.use_mode_weights is not None else cfg_use_mode_weights
    )
    return TrainingConfig(
        architecture_name=meta.architecture_name,
        model_name=model_name,
        model_params=meta.model_params,
        step=meta.step,
        shift=effective_shift,
        lr=ov.lr or meta.lr,
        weight_decay=1e-4,
        seq_len=effective_seq_len,
        batch_size=ov.batch_size or meta.batch_size,
        epochs=ov.epochs or 200,
        method=ov.method or meta.method,
        num_workers=4,
        lambda_tracking=ov.lambda_tracking or 0.0,
        alpha_dict={"fdm_heading_rad": 1.0},
        huber_beta_per_col={
            "raw_alt_m": 5.18e-2,
            "fdm_gamma_rad": 2.09e-1,
            "era_tas_ms": 1.02e-1,
        },
        eta_min=1e-5,
        use_mode_weights=use_mode_weights,
        mode_weight_alpha=cfg_mode_weight_alpha,
    )


def run_resume(
    *,
    model: Path,
    config: Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    seq_len: int | None = None,
    shift: int | None = None,
    overwrite: bool = False,
    device: str = "cpu",
    method: str | None = None,
    model_name: str | None = None,
    lambda_tracking: float | None = None,
    reset_loss: bool = False,
    typecode: str | None = None,
    use_mode_weights: bool | None = None,
) -> None:
    """Resume training from a saved model checkpoint."""
    from node_fdm.loader import get_train_val_data
    from node_fdm.trainer import ODETrainer

    from node_fdm_pipeline.config import PipelineConfig

    meta = _load_meta(model)
    info = _resolve_arch_info(meta.architecture_name)

    cfg = PipelineConfig.from_yaml(config)
    typecode_suffix = typecode or model.name.removeprefix(f"{meta.architecture_name}_")
    data_df = _load_data(cfg.paths.resolve("delta_table"), typecode_suffix)

    overrides = _Overrides(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seq_len=seq_len,
        shift=shift,
        method=method,
        model_name=model_name,
        lambda_tracking=lambda_tracking,
        use_mode_weights=use_mode_weights,
    )
    training_config = _build_training_config(
        meta,
        model.name,
        overrides,
        cfg_use_mode_weights=cfg.training.use_mode_weights,
        cfg_mode_weight_alpha=cfg.training.mode_weight_alpha,
    )

    from node_fdm.training.weighting import boot_mode_weights

    # Boot mode weights BEFORE building the dataset so each window/sample
    # carries its per-timestep weight column.
    if training_config.use_mode_weights:
        data_df = boot_mode_weights(data_df, alpha=training_config.mode_weight_alpha)

    dx_col_names = [col for _, col in info.dx_cols]
    train_ds, val_ds = get_train_val_data(
        data_df=data_df,
        x_cols=info.x_cols,
        u_cols=info.u_cols,
        e_cols=info.e0_cols,
        e1_cols=info.e1_cols,
        dx_cols=dx_col_names,
        seq_len=training_config.seq_len,
        shift=training_config.shift,
        train_limit=5000,
        val_limit=5000,
    )

    models_dir = model.parent if overwrite else cfg.paths.resolve("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "resume_start",
        model=str(model),
        architecture=meta.architecture_name,
        epochs=training_config.epochs,
        overwrite=overwrite,
        device=device,
    )

    trainer = ODETrainer(
        config=training_config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_dir=models_dir,
        device=device,
    )
    trainer.load_model_weights(reset_loss=reset_loss)
    trainer.load_optimizer_state()
    trainer.train()

    log.info("resume_done", model=str(model))
