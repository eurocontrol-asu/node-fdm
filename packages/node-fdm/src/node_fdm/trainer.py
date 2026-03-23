"""ODE trainer with Pydantic configuration and structured logging.

Replaces the legacy ``ODETrainer`` that used ``dict[str, Any]`` configs
and ``print()`` statements with a typed :class:`TrainingConfig` and
``structlog`` logging.
"""

from __future__ import annotations

import csv
import json
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
    grad_clip_norm: float = Field(default=1.0, gt=0)
    alpha_dict: dict[str, float] | None = None


def _collate_flight_samples(
    batch: list[FlightSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack flight samples into batched tensors."""
    return (
        torch.stack([s.x for s in batch]),
        torch.stack([s.u for s in batch]),
        torch.stack([s.e for s in batch]),
        torch.stack([s.dx for s in batch]),
    )


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

        # Compute stats from training data
        dx_col_names = [col for _, col in self.spec.dx_cols]
        self.stats_dict = compute_stats(
            list(train_dataset),  # type: ignore[call-overload]
            x_cols=self.spec.x_cols,
            u_cols=self.spec.u_cols,
            e_cols=self.spec.e0_cols,
            dx_cols=dx_col_names,
        )

        self.model = FlightDynamicsModel(self.spec, self.stats_dict, config.model_params).to(
            self.device
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.best_val_loss = float("inf")
        self.loss_fn: nn.Module = get_loss(config.loss_name)

        # Precompute normalization vectors for ODE rollout loss
        self._norm_mean, self._norm_std = self._build_norm_vectors()
        self._alpha_weights = self._build_alpha_weights()

        self.save_meta()
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
            stats = self.stats_dict.get(col, {"mean": 0.0, "std": 1.0})
            means.append(stats["mean"])
            stds.append(stats["std"])
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

    def _compute_batch_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss via ODE rollout trajectory comparison.

        Integrates the Neural ODE forward from ``x0`` using
        ``torchdiffeq.odeint`` and compares the predicted trajectory
        against the true state sequence. Both trajectories are
        normalized using dataset statistics and weighted by
        ``alpha_dict`` before the loss function is applied.

        Args:
            batch: Tuple of ``(x_seq, u_seq, e_seq, dx_seq)``.

        Returns:
            Scalar loss tensor.
        """
        x_seq, u_seq, e_seq, _ = (t.to(self.device) for t in batch)
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
        func = BatchNeuralODE(self.model, u_seq, e_seq, t_grid)
        x_pred = odeint(func, x0, t_grid, method=self.config.method)

        # odeint returns (time, batch, n_x) → (batch, time, n_x)
        x_pred = x_pred.permute(1, 0, 2)

        # Compare predicted vs true trajectory (skip initial condition)
        pred = x_pred[:, 1:, :]
        true = x_seq[:, 1:, :]

        # Normalize in loss space (scale-invariant across variables)
        pred_norm = (pred - self._norm_mean) / self._norm_std
        true_norm = (true - self._norm_mean) / self._norm_std

        # Apply per-variable alpha weights
        pred_weighted = pred_norm * self._alpha_weights
        true_weighted = true_norm * self._alpha_weights

        loss: torch.Tensor = self.loss_fn(pred_weighted, true_weighted)

        if torch.isnan(loss) or torch.isinf(loss):
            log.warning("nan_or_inf_loss", loss=loss.item())

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
