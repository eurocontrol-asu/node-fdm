"""ODE trainer with Pydantic configuration and structured logging.

Replaces the legacy ``ODETrainer`` that used ``dict[str, Any]`` configs
and ``print()`` statements with a typed :class:`TrainingConfig` and
``structlog`` logging.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import structlog
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from torchdiffeq import odeint

from node_fdm.architectures.registry import ArchitectureSpec, get
from node_fdm.callbacks import ConsoleCallback, TrainingCallback
from node_fdm.dataset import FlightDataset, FlightSample, compute_stats
from node_fdm.layers.mass_encoder import MassEncoderLinear
from node_fdm.losses import get_loss
from node_fdm.models.batch_neural_ode import BatchNeuralODE
from node_fdm.models.fdm import FlightDynamicsModel
from node_fdm.models.projected_integrator import (
    ClampedEuler,
    ClampedRK4,
    _clamp_columns,
)
from node_fdm.training.weighting import compute_segment_weights
from node_fdm_data.schemas.adsb_hybrid import (
    A320_MTOW_KG,
    A320_OEW_KG,
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
        eta_min: Minimum learning rate at end of decay. When ``None``
            (default) no scheduler is used and the lr stays at ``lr`` for
            the whole run.
        schedule: Decay shape after warmup. ``"linear"`` (default,
            recommended for sequential/regression tasks per Bergsma 2024
            "Straight to Zero") or ``"cosine"`` (legacy).
        warmup_epochs: Number of epochs to linearly ramp the lr from
            ``warmup_start_lr`` to ``lr``. ``0`` disables warmup.
        warmup_start_lr: Initial lr at the start of the warmup ramp.
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
    schedule: str = Field(default="linear", pattern="^(linear|cosine)$")
    warmup_epochs: int = Field(default=5, ge=0)
    warmup_start_lr: float = Field(default=1e-5, gt=0)
    use_mode_weights: bool = False
    mode_weight_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    activation: str = Field(default="silu", pattern="^(silu|relu|gelu|tanh)$")
    seed: int | None = Field(default=None, ge=0)


def _collate_flight_samples(
    batch: list[FlightSample],
) -> tuple[torch.Tensor, ...]:
    """Stack flight samples into batched tensors.

    Returns a 4-tuple ``(x, u, e, dx)`` when no ``e1`` data is present,
    or a 5-tuple ``(x, u, e, dx, e1)`` when samples carry tracking targets.
    """
    base: tuple[torch.Tensor, ...] = (
        torch.stack([s.x for s in batch]),
        torch.stack([s.u for s in batch]),
        torch.stack([s.e for s in batch]),
        torch.stack([s.dx for s in batch]),
    )
    if batch[0].e1 is not None:
        e1_stack = torch.stack([s.e1 for s in batch if s.e1 is not None])
        base = (*base, e1_stack)
    if batch[0].w is not None:
        w_stack = torch.stack([s.w for s in batch if s.w is not None])
        base = (*base, w_stack)
    if batch[0].flight_features is not None:
        ff_stack = torch.stack([s.flight_features for s in batch if s.flight_features is not None])
        base = (*base, ff_stack)
    return base


class ODETrainer:
    """Train a Neural ODE flight dynamics model.

    Uses :class:`TrainingConfig` for typed hyperparameters and
    ``structlog`` for all logging. Supports pluggable callbacks
    via the :class:`TrainingCallback` protocol.

    Args:
        config: Training configuration.
        train_dataset: Training dataset. When ``config.use_mode_weights`` is
            true, the caller is responsible for enriching the source
            DataFrame with ``fdm_train_weight`` (via
            :func:`node_fdm.training.weighting.boot_mode_weights`) **before**
            building the dataset, so each :class:`FlightSample` carries its
            per-timestep weight tensor ``w``.
        val_dataset: Validation dataset.
        model_dir: Base directory for checkpoints and metadata.
        callbacks: Optional list of training callbacks.
        device: Torch device string.
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

        # Seed the global RNGs as early as possible so that all downstream
        # randomness (weight init, shuffle, dropout, CUDA) shares the same
        # source. Default (``None``) preserves prior non-deterministic
        # behavior; the DataLoader generator below stays at seed 0 only
        # when no explicit seed is requested.
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

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
        flight_feature_cols = list(getattr(self.spec, "flight_feature_cols", []) or [])
        flight_feature_signs = list(getattr(self.spec, "flight_feature_signs", []) or [])
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
            flight_feature_cols=flight_feature_cols,
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
            flight_feature_cols=flight_feature_cols,
            derived_cols=derived_output_cols,
            derived_scale_floor_ratio=scale_floor,
        )

        self.model = FlightDynamicsModel(
            self.spec,
            model_stats,
            config.model_params,
            activation=config.activation,
        ).to(self.device)
        if self.device.type == "cuda":
            self.model = cast(FlightDynamicsModel, torch.compile(self.model))

        self.mass_encoder: MassEncoderLinear | None = None
        if flight_feature_cols:
            self.mass_encoder = MassEncoderLinear(
                feature_stats=self.stats_dict,
                feature_cols=flight_feature_cols,
                expected_signs=flight_feature_signs,
                oew_kg=A320_OEW_KG,
                mtow_kg=A320_MTOW_KG,
            ).to(self.device)
            if "fdm_mass_kg" in self.spec.x_cols and (
                config.alpha_dict is None or "fdm_mass_kg" not in config.alpha_dict
            ):
                log.warning(
                    "mass_dim_unweighted",
                    message=(
                        "fdm_mass_kg is in spec.x_cols but missing from "
                        "alpha_dict; set alpha_dict['fdm_mass_kg']=0.0 to "
                        "silence the residual on the mass dim."
                    ),
                )

        if self.mass_encoder is not None:
            self.optimizer = torch.optim.AdamW(
                list(self.model.parameters()) + list(self.mass_encoder.parameters()),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        self._override_m0_factor: float | None = None
        # Scheduler is built in ``train()`` once we know the number of
        # batches per epoch (step-wise scheduling).
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
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
        _pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=_collate_flight_samples,
            generator=torch.Generator().manual_seed(self.config.seed or 0),
            pin_memory=_pin,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_flight_samples,
            pin_memory=_pin,
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
        has_mass_encoder = getattr(self, "mass_encoder", None) is not None
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
            "activation": self.config.activation,
            "seed": self.config.seed,
            "epochs": self.config.epochs,
            "use_mode_weights": self.config.use_mode_weights,
            "mode_weight_alpha": self.config.mode_weight_alpha,
            "mass_encoder": has_mass_encoder,
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
        if self.mass_encoder is not None:
            torch.save(
                self.mass_encoder.state_dict(),
                self.model_dir / "mass_encoder.pt",
            )
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
        if self.mass_encoder is not None:
            mass_ckpt = self.model_dir / "mass_encoder.pt"
            if mass_ckpt.exists():
                state = torch.load(mass_ckpt, weights_only=True)
                self.mass_encoder.load_state_dict(state)
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
            # Synthetic state dims (e.g. fdm_mass_kg) are not observed in the
            # dataset — their data-driven stats collapse to ~0 and divide-by-
            # zero would blow up the rollout loss. Derive (mean, std) from the
            # spec's x_bounds when available: midpoint and half-range as a
            # reasonable scale.
            if col == "fdm_mass_kg" and col in self.spec.x_bounds:
                lo, hi = self.spec.x_bounds[col]
                means.append(0.5 * (lo + hi))
                stds.append(max((hi - lo) / 2.0, 1e-6))
                continue
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
        # The mass state dim is constant per segment by construction
        # (dm/dt = 0). Auto-zero its loss weight unless the user explicitly
        # overrides it via alpha_dict.
        if "fdm_mass_kg" in self.spec.x_cols and (
            self.config.alpha_dict is None or "fdm_mass_kg" not in self.config.alpha_dict
        ):
            mass_idx = self.spec.x_cols.index("fdm_mass_kg")
            weights[mass_idx] = 0.0
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

    @staticmethod
    def _preload_dataset(
        dataset: FlightDataset,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Collate an entire dataset into stacked tensors on *device*.

        Returns the same tuple shape as ``_collate_flight_samples`` but
        covering all samples, resident on *device*.
        """
        all_samples = [dataset[i] for i in range(len(dataset))]
        stacked = _collate_flight_samples(all_samples)
        return tuple(t.to(device) for t in stacked)

    @staticmethod
    def _iter_preloaded(
        stacked: tuple[torch.Tensor, ...],
        batch_size: int,
        shuffle: bool = True,
    ) -> list[tuple[torch.Tensor, ...]]:
        """Yield batch-sized slices from pre-loaded stacked tensors."""
        n = stacked[0].shape[0]
        if shuffle:
            perm = torch.randperm(n, device=stacked[0].device)
            stacked = tuple(t[perm] for t in stacked)
        batches = []
        for start in range(0, n, batch_size):
            end = start + batch_size
            batches.append(tuple(t[start:end] for t in stacked))
        return batches

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
        if batch[0].device == self.device:
            tensors = batch
        else:
            tensors = tuple(t.to(self.device, non_blocking=True) for t in batch)
        x_seq, u_seq, e_seq = tensors[0], tensors[1], tensors[2]
        # The collate output may carry e1, w and/or flight_features in trailing
        # slots. ``w`` has shape (batch, seq_len) → ndim==2. ``e1`` and
        # ``flight_features`` both have ndim==3; we identify ``flight_features``
        # by its trailing-dim against ``spec.flight_feature_cols``.
        w_tensor: torch.Tensor | None = None
        flight_features: torch.Tensor | None = None
        n_features = len(getattr(self.spec, "flight_feature_cols", []) or [])
        for t in tensors[4:]:
            if t.ndim == 2:
                w_tensor = t
            elif n_features > 0 and t.shape[-1] == n_features:
                flight_features = t

        seq_len = x_seq.shape[1]
        x0 = x_seq[:, 0, :]
        if self.mass_encoder is not None and flight_features is not None:
            m_0 = self.mass_encoder(flight_features[:, 0, :])
            if self._override_m0_factor is not None:
                m_0 = m_0 * self._override_m0_factor
            x0 = torch.cat([x0[:, :4], m_0.to(x0.dtype).unsqueeze(-1)], dim=-1)

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
            if w_tensor is None:
                loss: torch.Tensor = self.loss_fn(
                    residual_weighted,
                    torch.zeros_like(residual_weighted),
                )
            else:
                # Per-segment MSE then weight then mean over batch.
                # Note: the spec keeps the trajectory-loss reweighting only;
                # lambda_tracking term below is intentionally unweighted.
                per_segment = (residual_weighted**2).mean(dim=(1, 2))
                w_seg = compute_segment_weights(w_tensor)
                loss = (w_seg * per_segment).mean()
        else:
            # Per-column SmoothL1 with calibrated betas where provided;
            # MSE on columns with NaN beta (no override). Result averaged
            # across all elements to match the previous reduction='mean'.
            target_zero = torch.zeros_like(residual_weighted)
            if w_tensor is None:
                per_col_means = []
                for i in range(residual_weighted.shape[-1]):
                    col_res = residual_weighted[..., i]
                    col_zero = target_zero[..., i]
                    beta = self._huber_beta_per_col[i]
                    if torch.isnan(beta):
                        col_loss = torch.nn.functional.mse_loss(
                            col_res, col_zero, reduction="mean"
                        )
                    else:
                        col_loss = torch.nn.functional.smooth_l1_loss(
                            col_res, col_zero, reduction="mean", beta=beta.item()
                        )
                    per_col_means.append(col_loss)
                loss = torch.stack(per_col_means).mean()
            else:
                # Per-segment per-column reduction with reduction='none', mean over
                # (seq_len, n_x) → (batch,), multiply by w_seg, mean over batch.
                per_col_segment = []
                for i in range(residual_weighted.shape[-1]):
                    col_res = residual_weighted[..., i]
                    col_zero = target_zero[..., i]
                    beta = self._huber_beta_per_col[i]
                    if torch.isnan(beta):
                        elem = torch.nn.functional.mse_loss(col_res, col_zero, reduction="none")
                    else:
                        elem = torch.nn.functional.smooth_l1_loss(
                            col_res, col_zero, reduction="none", beta=beta.item()
                        )
                    per_col_segment.append(elem.mean(dim=1))
                per_segment = torch.stack(per_col_segment, dim=-1).mean(dim=-1)
                w_seg = compute_segment_weights(w_tensor)
                loss = (w_seg * per_segment).mean()

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

    def _build_scheduler(
        self, steps_per_epoch: int
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build a step-wise LR scheduler: linear warmup + (linear|cosine) decay.

        Returns ``None`` when ``eta_min`` is not set (constant lr).
        """
        cfg = self.config
        if cfg.eta_min is None:
            return None

        warmup_steps = max(cfg.warmup_epochs, 0) * steps_per_epoch
        total_steps = cfg.epochs * steps_per_epoch
        decay_steps = max(total_steps - warmup_steps, 1)

        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        milestones: list[int] = []
        if warmup_steps > 0:
            start_factor = cfg.warmup_start_lr / cfg.lr
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
            )
            milestones.append(warmup_steps)

        end_factor = cfg.eta_min / cfg.lr
        decay: torch.optim.lr_scheduler.LRScheduler
        if cfg.schedule == "linear":
            decay = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=decay_steps,
            )
        else:
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=decay_steps,
                eta_min=cfg.eta_min,
            )
        schedulers.append(decay)

        if len(schedulers) == 1:
            return schedulers[0]
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )

    def train(self) -> list[dict[str, float]]:
        """Run the full training loop.

        Returns:
            List of per-epoch loss records.
        """
        use_cuda = self.device.type == "cuda"

        if use_cuda:
            log.info("preloading_data_to_gpu")
            train_stacked = self._preload_dataset(self.train_dataset, self.device)
            val_stacked = self._preload_dataset(self.val_dataset, self.device)
            n_train_batches = max(
                (len(self.train_dataset) + self.config.batch_size - 1) // self.config.batch_size, 1
            )
        else:
            train_stacked = None
            val_stacked = None

        if not use_cuda:
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

        if use_cuda:
            steps_per_epoch = n_train_batches
        else:
            steps_per_epoch = max(len(train_loader), 1)
        self.scheduler = self._build_scheduler(steps_per_epoch=steps_per_epoch)

        for epoch in range(1, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_start(epoch, epochs)

            # --- Train ---
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            train_batches: Iterable[tuple[torch.Tensor, ...]]
            if use_cuda:
                assert train_stacked is not None
                train_batches = self._iter_preloaded(
                    train_stacked,
                    self.config.batch_size,
                    shuffle=True,
                )
            else:
                train_batches = cast(Iterable[tuple[torch.Tensor, ...]], train_loader)
            for batch in train_batches:
                loss = self._compute_batch_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                for p in self.model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                total_loss += loss.item()
                n_batches += 1
            avg_train = total_loss / max(n_batches, 1)

            # --- Validate ---
            self.model.eval()
            val_total = 0.0
            val_batches = 0
            val_iter: Iterable[tuple[torch.Tensor, ...]]
            if use_cuda:
                assert val_stacked is not None
                val_iter = self._iter_preloaded(
                    val_stacked,
                    self.config.val_batch_size,
                    shuffle=False,
                )
            else:
                val_iter = cast(Iterable[tuple[torch.Tensor, ...]], val_loader)
            with torch.no_grad():
                for batch in val_iter:
                    loss = self._compute_batch_loss(batch)
                    val_total += loss.item()
                    val_batches += 1
            avg_val = val_total / max(val_batches, 1)

            is_best = avg_val < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val
                self.save_model(epoch)

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

            if self.mass_encoder is not None:
                log.info(
                    "mass_encoder_coefs",
                    epoch=epoch,
                    **self.mass_encoder.effective_coefficients(),
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

    def identifiability_test(self, factor: float = 1.3) -> dict[str, float]:
        """Run the §11.1 identifiability gate.

        Computes baseline val MSE, then re-runs the validation rollout with
        ``m_0`` scaled by *factor* (no gradient) and reports the ratio. When
        ``self.mass_encoder is None``, both MSEs equal 1.0 (the perturbation
        has no effect because mass is taken from the data).

        Args:
            factor: Multiplicative perturbation applied to the predicted
                initial mass on the validation pass.

        Returns:
            ``{"baseline_mse": ..., "perturbed_mse": ..., "ratio": ...}``.
        """
        if self.mass_encoder is None:
            return {"baseline_mse": 1.0, "perturbed_mse": 1.0, "ratio": 1.0}

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_flight_samples,
        )

        def _mean_loss(override: float | None) -> float:
            self._override_m0_factor = override
            total = 0.0
            n = 0
            try:
                with torch.no_grad():
                    for batch in val_loader:
                        loss = self._compute_batch_loss(batch)
                        total += float(loss.item())
                        n += 1
            finally:
                self._override_m0_factor = None
            return total / max(n, 1)

        self.model.eval()
        baseline_mse = _mean_loss(None)
        perturbed_mse = _mean_loss(factor)
        ratio = perturbed_mse / baseline_mse if baseline_mse > 0 else float("inf")
        return {
            "baseline_mse": baseline_mse,
            "perturbed_mse": perturbed_mse,
            "ratio": ratio,
        }
