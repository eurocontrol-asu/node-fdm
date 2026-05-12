"""Training command — ``fdm train``.

Reads flight data from the Delta Table, filters on validity flags,
and trains Neural ODE models per aircraft typecode.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import polars as pl

__all__ = ["run_training"]

log = structlog.get_logger()


@dataclass(frozen=True)
class _TrainOverrides:
    """CLI override values applied on top of the default training config."""

    epochs: int | None
    batch_size: int | None
    lr: float | None
    method: str
    seq_len: int | None
    shift: int | None
    model_name: str | None
    lambda_tracking: float | None
    use_mode_weights: bool | None
    train_limit: int | None


@dataclass(frozen=True)
class _TrainContext:
    """Bundle of resolved inputs shared across per-typecode training calls."""

    info: Any
    full_df: pl.DataFrame
    dx_col_names: list[str]
    models_dir: Path
    device: str
    overrides: _TrainOverrides
    cfg_use_mode_weights: bool = False


def _load_delta_df(cfg: Any) -> tuple[Any, Path]:
    """Read the valid-rows Delta table and return it alongside the resolved models_dir."""
    import polars as pl

    delta_table = cfg.paths.resolve("delta_table")
    if not delta_table.exists():
        log.error("train_missing_delta", path=str(delta_table))
        msg = f"Delta table not found at {delta_table}. Run the pipeline first."
        raise SystemExit(msg)

    models_dir = cfg.paths.resolve("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    full_df = pl.read_delta(str(delta_table)).filter(pl.col("fdm_flag_valid"))
    return full_df, models_dir


def _build_training_config(ctx: _TrainContext, acft: str) -> Any:
    """Build the per-typecode TrainingConfig with overrides and tuned defaults."""
    from node_fdm.trainer import TrainingConfig

    ov = ctx.overrides
    effective_seq_len = ov.seq_len or 60
    use_mode_weights = (
        ov.use_mode_weights if ov.use_mode_weights is not None else ctx.cfg_use_mode_weights
    )
    return TrainingConfig(
        architecture_name=ctx.info.name,
        model_name=ov.model_name or f"{ctx.info.name}_{acft}",
        model_params=(3, 2, 48),
        step=4.0,
        shift=ov.shift or effective_seq_len,
        lr=ov.lr or 5e-4,
        weight_decay=1e-4,
        seq_len=effective_seq_len,
        batch_size=ov.batch_size or 512,
        epochs=ov.epochs or 800,
        method=ov.method,
        num_workers=4,
        lambda_tracking=ov.lambda_tracking or 0.0,
        grad_clip_norm=10.0,
        alpha_dict={"fdm_heading_rad": 1.0},
        huber_beta_per_col={
            "raw_alt_m": 5.18e-2,
            "fdm_gamma_rad": 2.09e-1,
            "era_tas_ms": 1.02e-1,
        },
        eta_min=1e-5,
        use_mode_weights=use_mode_weights,
    )


def _maybe_adjust_epochs(
    training_config: Any,
    train_ds: Any,
    acft: str,
    epochs_override: int | None,
) -> Any:
    """Scale epochs by dataset size when no explicit override is given."""
    if epochs_override is not None:
        return training_config

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
    return training_config.model_copy(update={"epochs": adjusted_epochs})


def _train_one_typecode(ctx: _TrainContext, acft: str) -> None:
    """Train a Neural ODE model for one typecode end-to-end (data → config → fit)."""
    import polars as pl
    from node_fdm.loader import get_train_val_data
    from node_fdm.trainer import ODETrainer

    log.info("train_typecode", typecode=acft)

    training_config = _build_training_config(ctx, acft)
    data_df = ctx.full_df.filter(pl.col("meta_aircraft_type") == acft)

    if len(data_df) == 0:
        log.warning("train_empty_dataset", typecode=acft)
        return

    train_ds, val_ds = get_train_val_data(
        data_df=data_df,
        x_cols=ctx.info.x_cols,
        u_cols=ctx.info.u_cols,
        e_cols=ctx.info.e0_cols,
        e1_cols=ctx.info.e1_cols,
        dx_cols=ctx.dx_col_names,
        seq_len=training_config.seq_len,
        shift=training_config.shift,
        train_limit=ctx.overrides.train_limit or 5000,
        val_limit=min(ctx.overrides.train_limit or 5000, 5000),
    )

    training_config = _maybe_adjust_epochs(training_config, train_ds, acft, ctx.overrides.epochs)

    trainer = ODETrainer(
        config=training_config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_dir=ctx.models_dir,
        device=ctx.device,
        train_df=data_df if training_config.use_mode_weights else None,
    )
    trainer.train()
    log.info("train_typecode_done", typecode=acft)


def run_training(
    *,
    arch: str,
    config: Path,
    typecode: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    method: str = "euler",
    seq_len: int | None = None,
    shift: int | None = None,
    device: str = "cpu",
    model_name: str | None = None,
    lambda_tracking: float | None = None,
    use_mode_weights: bool | None = None,
    train_limit: int | None = None,
) -> None:
    """Train Neural ODE models for one or all typecodes.

    Args:
        arch: Architecture identifier (``"qar"`` or ``"adsb"``).
        config: Path to YAML pipeline config.
        typecode: Single typecode to train (default: all from config).
        epochs: Override number of training epochs.
        batch_size: Override batch size.
        lr: Override learning rate.
        method: ODE integration method (``"euler"`` or ``"rk4"``).
        seq_len: Override sequence length for training windows.
        shift: Override shift between windows (defaults to seq_len).
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).
        model_name: Custom model name (default: ``{arch}_{typecode}``).
        train_limit: Max training samples (default: 5000).
        use_mode_weights: Optional CLI override. ``None`` defers to
            ``cfg.training.use_mode_weights``; otherwise the explicit value
            wins (precedence: CLI > YAML > default ``False``).
    """
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)
    importlib.import_module(info.architecture_import)

    typecodes = [typecode] if typecode else cfg.typecodes
    full_df, models_dir = _load_delta_df(cfg)

    ctx = _TrainContext(
        info=info,
        full_df=full_df,
        dx_col_names=[col for _, col in info.dx_cols],
        models_dir=models_dir,
        device=device,
        overrides=_TrainOverrides(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            method=method,
            seq_len=seq_len,
            shift=shift,
            model_name=model_name,
            lambda_tracking=lambda_tracking,
            use_mode_weights=use_mode_weights,
            train_limit=train_limit,
        ),
        cfg_use_mode_weights=cfg.training.use_mode_weights,
    )

    log.info("train_start", arch=arch, typecodes=typecodes, device=device)
    for acft in typecodes:
        _train_one_typecode(ctx, acft)
    log.info("train_done", typecodes=typecodes)
