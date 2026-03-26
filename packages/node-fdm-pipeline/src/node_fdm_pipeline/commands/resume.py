"""Resume training from an existing checkpoint."""

from __future__ import annotations

import importlib
from pathlib import Path

import structlog

from node_fdm_pipeline.resolver import ARCH_BY_NAME, resolve_architecture

__all__ = ["run_resume"]

log = structlog.get_logger("node_fdm_pipeline.commands.resume")


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
    lambda_tracking: float | None = None,
) -> None:
    """Resume training from a saved model checkpoint.

    Loads ``meta.json`` from *model*, infers the architecture, rebuilds
    datasets, and continues training.  CLI overrides (epochs, lr, etc.)
    take precedence over the values stored in the checkpoint.

    Args:
        model: Path to the model directory containing ``meta.json``.
        config: Path to YAML pipeline config.
        epochs: Override number of training epochs.
        batch_size: Override batch size.
        lr: Override learning rate.
        seq_len: Override sequence length for training windows.
        shift: Override shift between windows.
        overwrite: Save back into the same model directory.
        device: PyTorch device string.
        method: Override ODE integration method (``"euler"`` or ``"rk4"``).
        lambda_tracking: Tracking loss weight (0=disabled).
    """
    import polars as pl
    from node_fdm.loader import get_train_val_data
    from node_fdm.predictor import ModelMeta
    from node_fdm.trainer import ODETrainer, TrainingConfig

    from node_fdm_pipeline.config import PipelineConfig

    # --- Load meta --------------------------------------------------------
    meta_path = model / "meta.json"
    if not meta_path.exists():
        log.error("resume_missing_meta", path=str(meta_path))
        msg = f"meta.json not found at {meta_path}"
        raise SystemExit(msg)

    meta = ModelMeta.from_json(meta_path)

    # --- Resolve architecture from meta -----------------------------------
    arch_key = ARCH_BY_NAME.get(meta.architecture_name)
    if arch_key is None:
        msg = (
            f"Unknown architecture_name in meta.json: {meta.architecture_name!r}. "
            f"Known: {', '.join(sorted(ARCH_BY_NAME))}"
        )
        raise SystemExit(msg)

    info = resolve_architecture(arch_key)
    importlib.import_module(info.architecture_import)

    # --- Pipeline config & data -------------------------------------------
    cfg = PipelineConfig.from_yaml(config)

    delta_table = cfg.paths.resolve("delta_table")
    if not delta_table.exists():
        log.error("resume_missing_delta", path=str(delta_table))
        msg = f"Delta table not found at {delta_table}. Run the pipeline first."
        raise SystemExit(msg)

    # Infer typecode from model_name  (e.g. "opensky_2025_A320" → "A320")
    model_name = model.name
    typecode_suffix = model_name.removeprefix(f"{meta.architecture_name}_")

    full_df = pl.read_delta(str(delta_table))
    full_df = full_df.filter(pl.col("fdm_flag_valid"))
    data_df = full_df.filter(pl.col("meta_aircraft_type") == typecode_suffix)

    if len(data_df) == 0:
        log.warning("resume_empty_dataset", typecode=typecode_suffix)
        msg = f"No data for typecode {typecode_suffix!r}."
        raise SystemExit(msg)

    # --- Build TrainingConfig with overrides ------------------------------
    effective_seq_len = seq_len or meta.seq_len
    effective_shift = shift or effective_seq_len

    training_config = TrainingConfig(
        architecture_name=meta.architecture_name,
        model_name=model_name,
        model_params=meta.model_params,
        step=meta.step,
        shift=effective_shift,
        lr=lr or meta.lr,
        weight_decay=1e-4,
        seq_len=effective_seq_len,
        batch_size=batch_size or meta.batch_size,
        epochs=epochs or 200,
        method=method or meta.method,
        num_workers=4,
        lambda_tracking=lambda_tracking or 0.0,
    )

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

    # --- Determine output directory ---------------------------------------
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
    trainer.load_model_weights()
    trainer.load_optimizer_state()
    trainer.train()

    log.info("resume_done", model=str(model))
