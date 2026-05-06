"""ODE trainer with Pydantic configuration and structured logging.

Replaces the legacy ``ODETrainer`` that used ``dict[str, Any]`` configs
and ``print()`` statements with a typed :class:`TrainingConfig` and
``structlog`` logging.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from torchdiffeq import odeint

from node_fdm.architectures.registry import ArchitectureSpec, get
from node_fdm.callbacks import ConsoleCallback, TrainingCallback
from node_fdm.dataset import FlightDataset, FlightSample, compute_stats
from node_fdm.losses import get_loss
from node_fdm.models.batch_neural_ode import BatchNeuralODE
from node_fdm.models.fdm import FlightDynamicsModel
from node_fdm.models.projected_integrator import (
    ClampedEuler,
    ClampedRK4,
    _clamp_columns,
)

__all__ = [
    "ODETrainer",
    "TrainingConfig",
]

log = structlog.get_logger("node_fdm.trainer")


class TrainingConfig(BaseModel):
    """Typed training hyperparameters.

    Replaces the legacy ``model_config: dict[str, Any]`` with validated,
    documented fields.

    Attributes:
        architecture_name: Name of the registered architecture spec.
        model_name: Identifier for saving checkpoints.
        model_params: Tuple of ``(backbone_depth, head_depth, hidden_width)``.
        seq_len: Sequence length for windowed samples.
        shift: Window shift between consecutive samples.
        step: Integration timestep for the ODE solver.
        lr: Learning rate.
        weight_decay: L2 regularization factor.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        method: ODE solver method (e.g. ``"rk4"``).
        val_batch_size: Validation batch size.
        num_workers: Number of DataLoader workers.
        loss_name: Loss function identifier.
        grad_clip_norm: Max gradient norm for clipping.
        alpha_dict: Per-variable loss weighting for ``x_cols``.
            Defaults to ``1.0`` for all variables when ``None``.
        eta_min: Minimum learning rate for ``CosineAnnealingLR``. When
            ``None`` (default) no scheduler is used and the lr stays at
            ``lr`` for the whole run.
    """

    architecture_name: str
    model_name: str
    model_params: tuple[int, int, int] = (2, 1, 48)
    seq_len: int = 60
    shift: int = 60
    step: float = 1.0
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    batch_size: int = Field(default=512, gt=0)
    epochs: int = Field(default=800, gt=0)
    method: str = "rk4"
    val_batch_size: int = Field(default=10000, gt=0)
    num_workers: int = Field(default=4, ge=0)
    loss_name: str = "mse"
    grad_clip_norm: float = Field(default=10.0, gt=0)
    alpha_dict: dict[str, float] | None = None
    lambda_tracking: float = Field(default=0.0, ge=0)
    huber_beta_per_col: dict[str, float] | None = None
    eta_min: float | None = None


def _collate_flight_samples(
    batch: list[FlightSample],
) -> tuple[torch.Tensor, ...]:
    """Stack flight samples into batched tensors.

    Returns a 4-tuple ``(x, u, e, dx)`` when no ``e1`` data is present,
    or a 5-tuple ``(x, u, e, dx, e1)`` when samples carry tracking targets.
    """
    base = (
        torch.stack([s.x for s in batch]),
        torch.stack([s.u for s in batch]),
        torch.stack([s.e for s in batch]),
        torch.stack([s.dx for s in batch]),
    )
    if batch[0].e1 is not None:
        return (*base, torch.stack([s.e1 for s in batch]))  # type: ignore[misc]
    return base


class ODETrainer:
    """Train a Neural ODE flight dynamics model.

    Uses :class:`TrainingConfig` for typed hyperparameters and
    ``structlog`` for all logging. Supports pluggable callbacks
    via the :class:`TrainingCallback` protocol.

    Args:
        config: Training configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        model_dir: Base directory for checkpoints and metadata.
        callbacks: Optional list of training callbacks.
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: FlightDataset,
        val_dataset: FlightDataset,
        model_dir: Path,
        callbacks: Sequence[TrainingCallback] | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        self.spec: ArchitectureSpec = get(config.architecture_name)
        self.model_dir = model_dir / config.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.callbacks: Sequence[TrainingCallback] = callbacks or [ConsoleCallback()]

        # Compute stats from training data.
        # U_ODE_COLS from the architecture spec (empty for adsb).
        # Do NOT reconstruct from input_cols — passthrough flags like
        # fdm_gamma_target_known appear in input_cols but are not ODE
        # controls and must not enter compute_stats (which indexes the
        # u tensor by position, causing column misalignment).
        u_ode_cols = list(getattr(self.spec, "u_ode_cols", []) or [])
        dx_col_names = [col for _, col in self.spec.dx_cols]
        e1_cols = self.spec.e1_cols if hasattr(self.spec, "e1_cols") else None
        derived_output_cols = list(getattr(self.spec, "derived_output_cols", []) or [])
        _samples = list(train_dataset)  # type: ignore[call-overload]
        _stats_args = {
            "x_cols": self.spec.x_cols,
            "u_cols": u_ode_cols,
            "e_cols": self.spec.e0_cols,
            "dx_cols": dx_col_names,
        }
        scale_floor = float(getattr(self.spec, "nn_output_scale_floor_ratio", 0.0) or 0.0)
        self.stats_dict = compute_stats(
            _samples,
            **_stats_args,
            e1_cols=e1_cols,
            derived_cols=derived_output_cols,
            derived_scale_floor_ratio=scale_floor,
        )

        # Model stats: include e1 so StructuredLayer inputs are normalized.
        # compute_stats skips e1 columns already covered by DX, so no
        # overwrite risk for overlapping columns like fdm_d_alt_ms.
        model_stats = compute_stats(
            _samples,
            **_stats_args,
            e1_cols=e1_cols,
            derived_cols=derived_output_cols,
            derived_scale_floor_ratio=scale_floor,
        )

        self.model = FlightDynamicsModel(self.spec, model_stats, config.model_params).to(
            self.device
        )

        # Freeze GammaDefaultNet when tracking loss is disabled — the ODE
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        if config.eta_min is not None:
            self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.epochs,
                    eta_min=config.eta_min,
                )
            )
        else:
            self.scheduler = None
        self.best_val_loss = float("inf")
        self.loss_fn: nn.Module = get_loss(config.loss_name)

        # Precompute normalization vectors for ODE rollout loss
        self._norm_mean, self._norm_std = self._build_norm_vectors()
        self._alpha_weights = self._build_alpha_weights()
        self._huber_beta_per_col = self._build_huber_betas()

        # Index of the heading state (lateral channel) — used by the
        # rollout loss to apply signed_wrap on the residual instead of the
        # raw difference.  ``None`` if the architecture has no heading.
        self._heading_idx: int | None = None
        if "fdm_heading_rad" in self.spec.x_cols:
            self._heading_idx = self.spec.x_cols.index("fdm_heading_rad")

        self.save_meta()

        # Deterministic loaders for reproducible single-batch access
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=_collate_flight_samples,
            generator=torch.Generator().manual_seed(0),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_flight_samples,
        )

        log.info(
            "trainer_initialized",
            architecture=config.architecture_name,
            device=str(self.device),
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
        )

    def save_meta(self) -> None:
        """Persist training metadata compatible with :class:`ModelMeta`."""
        optimizer_path = self.model_dir / "optimizer.pt"
        meta: dict[str, Any] = {
            "architecture_name": self.config.architecture_name,
            "model_params": list(self.config.model_params),
            "step": self.config.step,
            "shift": self.config.shift,
            "lr": self.config.lr,
            "seq_len": self.config.seq_len,
            "batch_size": self.config.batch_size,
            "method": self.config.method,
            "stats_dict": self.stats_dict,
            "optimizer_saved": optimizer_path.exists(),
        }
        meta_path = self.model_dir / "meta.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        log.debug("meta_saved", path=str(meta_path))

    def save_layer_checkpoint(self, layer_name: str, epoch: int) -> None:
        """Save checkpoint for a single layer.

        Args:
            layer_name: Layer identifier.
            epoch: Current epoch number.
        """
        layer = self.model.layers_dict[layer_name]
        save_dict = {
            "layer_state": layer.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epoch": epoch,
        }
        torch.save(save_dict, self.model_dir / f"{layer_name}.pt")

    def save_model(self, epoch: int) -> None:
        """Save checkpoints for all layers and optimizer state.

        Args:
            epoch: Current epoch number.
        """
        for name in self.model.layers_name:
            self.save_layer_checkpoint(name, epoch)
        torch.save(self.optimizer.state_dict(), self.model_dir / "optimizer.pt")
        self.save_meta()
        log.debug("model_saved", epoch=epoch)

    def load_model_weights(self, *, reset_loss: bool = False) -> None:
        """Load layer weights from checkpoints saved by :meth:`save_layer_checkpoint`.

        For each layer in ``model.layers_name``, loads the corresponding
        ``.pt`` file, extracts ``layer_state``, and restores it.  Also
        restores ``best_val_loss`` from the checkpoint unless *reset_loss*
        is ``True``, in which case ``best_val_loss`` stays at ``inf`` so
        that the first improving epoch triggers a save.

        Args:
            reset_loss: If ``True``, ignore the saved ``best_val_loss``
                and keep the initial ``inf`` value.

        Raises:
            FileNotFoundError: If a layer checkpoint file is missing.
        """
        for name in self.model.layers_name:
            ckpt_path = self.model_dir / f"{name}.pt"
            if not ckpt_path.exists():
                msg = f"Layer checkpoint not found: {ckpt_path}"
                raise FileNotFoundError(msg)
            ckpt = torch.load(ckpt_path, weights_only=True)
            self.model.layers_dict[name].load_state_dict(ckpt["layer_state"])
            if not reset_loss:
                self.best_val_loss = ckpt.get("best_val_loss", self.best_val_loss)
        log.debug(
            "model_weights_loaded",
            layers=list(self.model.layers_name),
            reset_loss=reset_loss,
        )

    def load_optimizer_state(self) -> None:
        """Load optimizer state from checkpoint if available.

        If ``optimizer.pt`` does not exist, logs a warning and returns
        without modifying the optimizer (fresh start).
        """
        optimizer_path = self.model_dir / "optimizer.pt"
        if not optimizer_path.exists():
            log.warning("optimizer_checkpoint_missing", path=str(optimizer_path))
            return
        state = torch.load(optimizer_path, weights_only=True)
        self.optimizer.load_state_dict(state)
        log.debug("optimizer_state_loaded", path=str(optimizer_path))

    def _build_norm_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build normalization mean/std tensors for ``x_cols``.

        Returns:
            Tuple of ``(mean, std)`` tensors of shape ``(n_x,)`` on
            ``self.device``.
        """
        means: list[float] = []
        stds: list[float] = []
        for col in self.spec.x_cols:
            stats = self.stats_dict.get(col, {"mean": 0.0, "std": 1.0, "iqr": 1.0})
            means.append(stats["mean"])
            # Use IQR 0.5-99.5 for loss normalization when available.
            # IQR is robust to the cruise-dominated distribution that
            # makes std too small for gamma (→ 100% of loss) and too
            # large for altitude (→ 0% of loss).
            stds.append(stats.get("iqr", stats["std"]) / 5.0)
        return (
            torch.tensor(means, device=self.device),
            torch.tensor(stds, device=self.device),
        )

    def _build_alpha_weights(self) -> torch.Tensor:
        """Build per-variable weight vector from ``alpha_dict``.

        Returns:
            Tensor of shape ``(n_x,)`` with per-variable weights.
        """
        n_x = len(self.spec.x_cols)
        weights = torch.ones(n_x, device=self.device)
        if self.config.alpha_dict is not None:
            for i, col in enumerate(self.spec.x_cols):
                if col in self.config.alpha_dict:
                    weights[i] = self.config.alpha_dict[col]
        return weights

    def _build_huber_betas(self) -> torch.Tensor | None:
        if self.config.huber_beta_per_col is None:
            return None
        n_x = len(self.spec.x_cols)
        betas = torch.full((n_x,), float("nan"), device=self.device)
        for i, col in enumerate(self.spec.x_cols):
            if col in self.config.huber_beta_per_col:
                betas[i] = float(self.config.huber_beta_per_col[col])
        return betas

    @staticmethod
    def _resolve_bounds(
        named_bounds: dict[str, tuple[float, float]],
        cols: list[str] | list[tuple[int, str]],
    ) -> dict[int, tuple[float, float]]:
        """Convert named bounds to column-index bounds.

        Args:
            named_bounds: Mapping from column name to ``(lo, hi)``.
            cols: Column list — either ``["name", ...]`` for x_cols or
                ``[(sign, "name"), ...]`` for dx_cols.

        Returns:
            Mapping from column index to ``(lo, hi)``.
        """
        if not named_bounds:
            return {}
        result: dict[int, tuple[float, float]] = {}
        for i, col in enumerate(cols):
            name: str = col[1] if isinstance(col, tuple) else col  # type: ignore[assignment]
            if name in named_bounds:
                result[i] = named_bounds[name]
        return result

    def _compute_batch_loss(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute loss via ODE rollout trajectory comparison.

        Integrates the Neural ODE forward from ``x0`` using
        ``torchdiffeq.odeint`` and compares the predicted trajectory
        against the true state sequence. Both trajectories are
        normalized using dataset statistics and weighted by
        ``alpha_dict`` before the loss function is applied.

        When ``lambda_tracking > 0`` and the batch contains an e1 tensor
        (tracking targets), a masked MSE tracking term is added to the
        ODE rollout loss.  The ``known`` mask is derived from finite
        values in e1 (NaN = unknown / autopilot disengaged).

        Args:
            batch: Tuple of ``(x_seq, u_seq, e_seq, dx_seq)`` or
                ``(x_seq, u_seq, e_seq, dx_seq, e1_seq)``.

        Returns:
            Scalar loss tensor.
        """
        tensors = tuple(t.to(self.device) for t in batch)
        x_seq, u_seq, e_seq = tensors[0], tensors[1], tensors[2]
        _ = tensors[4] if len(tensors) == 5 else None  # e1_seq reserved for future use

        seq_len = x_seq.shape[1]
        x0 = x_seq[:, 0, :]

        t_grid = torch.arange(
            0,
            seq_len * self.config.step,
            self.config.step,
            dtype=torch.float32,
            device=self.device,
        )

        self.model.reset_history()

        # Convert named bounds to column-index dicts
        x_bounds_idx = self._resolve_bounds(self.spec.x_bounds, self.spec.x_cols)
        dx_bounds_idx = self._resolve_bounds(self.spec.dx_bounds, self.spec.dx_cols)

        func = BatchNeuralODE(self.model, u_seq, e_seq, t_grid, dx_bounds=dx_bounds_idx)

        if x_bounds_idx:
            # Use projected integrator for bounded specs
            project_fn = lambda x: _clamp_columns(x, x_bounds_idx)  # noqa: E731
            method = self.config.method
            solver_kwargs = {
                "atol": 1e-6,
                "rtol": 1e-3,
                "step_size": self.config.step,
            }
            if method == "euler":
                solver = ClampedEuler(func, x0, project_fn=project_fn, **solver_kwargs)
                x_pred = solver.integrate(t_grid)
            elif method == "rk4":
                solver = ClampedRK4(func, x0, project_fn=project_fn, **solver_kwargs)
                x_pred = solver.integrate(t_grid)
            else:
                msg = (
                    f"Method '{method}' does not support state projection. "
                    f"Use 'euler' or 'rk4' with x_bounds."
                )
                raise ValueError(msg)
        else:
            x_pred = odeint(func, x0, t_grid, method=self.config.method)

        # odeint / solver returns (time, batch, n_x) → (batch, time, n_x)
        x_pred = x_pred.permute(1, 0, 2)

        # Compare predicted vs true trajectory (skip initial condition)
        pred = x_pred[:, 1:, :]
        true = x_seq[:, 1:, :]

        # Wrap-aware residual for the heading state (lateral channel).
        # The heading is integrated freely (no x_bounds clamp) and can
        # drift past 2π over long rollouts; without signed_wrap, a
        # physically correct prediction one full turn ahead would yield
        # a catastrophic MSE of ~(2π)² ≈ 39.5.  See
        # ``scripts/debug/validate_lateral_wrap.py`` for the numerical
        # validation.  Mean cancels in (pred - true) so we work directly
        # with the residual and skip the per-column mean step that the
        # symmetric MSE would otherwise apply.
        residual = pred - true
        if self._heading_idx is not None:
            two_pi = 2.0 * math.pi
            raw = residual[..., self._heading_idx]
            wrapped = ((raw + math.pi) % two_pi) - math.pi
            residual = residual.clone()
            residual[..., self._heading_idx] = wrapped

        residual_norm = residual / self._norm_std
        residual_weighted = residual_norm * self._alpha_weights

        # Kept for the debug log below — defined before the conditional so
        # the magnitude trace can recover the un-wrapped form for parity
        # checks if needed.
        pred_norm = (pred - self._norm_mean) / self._norm_std
        true_norm = (true - self._norm_mean) / self._norm_std

        if self._huber_beta_per_col is None:
            loss: torch.Tensor = self.loss_fn(
                residual_weighted,
                torch.zeros_like(residual_weighted),
            )
        else:
            # Per-column SmoothL1 with calibrated betas where provided;
            # MSE on columns with NaN beta (no override). Result averaged
            # across all elements to match the previous reduction='mean'.
            target_zero = torch.zeros_like(residual_weighted)
            per_col_means = []
            for i in range(residual_weighted.shape[-1]):
                col_res = residual_weighted[..., i]
                col_zero = target_zero[..., i]
                beta = self._huber_beta_per_col[i]
                if torch.isnan(beta):
                    col_loss = torch.nn.functional.mse_loss(col_res, col_zero, reduction="mean")
                else:
                    col_loss = torch.nn.functional.smooth_l1_loss(
                        col_res, col_zero, reduction="mean", beta=beta.item()
                    )
                per_col_means.append(col_loss)
            loss = torch.stack(per_col_means).mean()

        # --- Tracking loss on autopilot targets ---
        # Compares predicted states with target consignes from U_COLS.
        # U_COLS order: [alt_target, tas_target, gamma_target, gamma_known, tas_known]
        # Positional indexing on indices 0..2 only — known flags ignored here.
        # X_COLS order: [alt, gamma, tas]
        if self.config.lambda_tracking > 0 and u_seq.shape[2] >= 4:
            u_skip = u_seq[:, 1:, :]  # skip initial condition
            pred = x_pred[:, 1:, :]

            alt_target = u_skip[:, :, 0]
            tas_target = u_skip[:, :, 1]
            gamma_target = u_skip[:, :, 2]

            alt_pred = pred[:, :, 0]  # raw_alt_m
            gamma_pred = pred[:, :, 1]  # fdm_gamma_rad
            tas_pred = pred[:, :, 2]  # era_tas_ms

            # MSE in z-score space (consistent with main ODE loss)
            std_alt = self._norm_std[0]
            std_gamma = self._norm_std[1]
            std_tas = self._norm_std[2]

            # All 3 targets always active:
            # - alt/tas: backfilled, always valid
            # - gamma: real target when known=1, gamma_default when known=0
            #   (both cases help the model learn)
            tracking_loss = (
                ((alt_target - alt_pred) / std_alt) ** 2
                + ((tas_target - tas_pred) / std_tas) ** 2
                + ((gamma_target - gamma_pred) / std_gamma) ** 2
            ).mean()

            loss = loss + self.config.lambda_tracking * tracking_loss

        if torch.isnan(loss) or torch.isinf(loss):
            log.warning("nan_or_inf_loss", loss=loss.item())

        # --- DEBUG: log magnitudes on first batch of first epoch ---
        if not getattr(self, "_debug_logged", False):
            self._debug_logged = True
            with torch.no_grad():
                log.info(
                    "debug_magnitudes",
                    norm_std=[f"{v:.6f}" for v in self._norm_std.tolist()],
                    norm_mean=[f"{v:.6f}" for v in self._norm_mean.tolist()],
                    ode_loss=(
                        f"{self.loss_fn(pred_norm, true_norm).item():.6f}"
                        if "pred_norm" in dir()
                        else "n/a"
                    ),
                )
                if self.config.lambda_tracking > 0 and u_seq.shape[2] >= 4:
                    alt_err = torch.abs(alt_target - alt_pred)
                    tas_err = torch.abs(tas_target - tas_pred)
                    gam_err = torch.abs(gamma_target - gamma_pred)
                    log.info(
                        "debug_tracking_errors",
                        alt_err_mean=f"{alt_err.mean().item():.4f}",
                        alt_err_max=f"{alt_err.max().item():.4f}",
                        tas_err_mean=f"{tas_err.mean().item():.4f}",
                        tas_err_max=f"{tas_err.max().item():.4f}",
                        gam_err_mean=f"{gam_err.mean().item():.4f}",
                        gam_err_max=f"{gam_err.max().item():.4f}",
                        tracking_loss=f"{tracking_loss.item():.4f}",
                        weighted_tracking=(
                            f"{(self.config.lambda_tracking * tracking_loss).item():.4f}"
                        ),
                    )

        return loss

    def train(self) -> list[dict[str, float]]:
        """Run the full training loop.

        Returns:
            List of per-epoch loss records.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=_collate_flight_samples,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_flight_samples,
        )

        epochs = self.config.epochs
        records: list[dict[str, float]] = []
        loss_csv_path = self.model_dir / "training_losses.csv"

        for epoch in range(1, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_start(epoch, epochs)

            # --- Train ---
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                loss = self._compute_batch_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                # Sanitize NaN/Inf gradients from ODE rollout through
                # physics layers before clipping and stepping.
                for p in self.model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_train = total_loss / max(n_batches, 1)

            # --- Validate ---
            self.model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = self._compute_batch_loss(batch)
                    val_total += loss.item()
                    val_batches += 1
            avg_val = val_total / max(val_batches, 1)

            is_best = avg_val < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val
                self.save_model(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": float(epoch),
                "train_loss": avg_train,
                "val_loss": avg_val,
            }
            records.append(record)

            for cb in self.callbacks:
                cb.on_epoch_end(
                    epoch,
                    epochs,
                    avg_train,
                    avg_val,
                    is_best=is_best,
                    lr=current_lr,
                )

        # Write loss CSV (no pandas)
        with loss_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(records)
        log.info("training_log_saved", path=str(loss_csv_path))

        for cb in self.callbacks:
            cb.on_train_end(self.best_val_loss)

        return records
