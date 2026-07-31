"""Prediction helper with typed model metadata.

Replaces the legacy ``NodeFDMPredictor`` that used ``dict[str, Any]``
column definitions and pandas DataFrames with Pydantic-typed
:class:`ModelMeta` and NumPy array outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from pydantic import BaseModel

from node_fdm.architectures.registry import get
from node_fdm.models.batch_neural_ode import BatchNeuralODE
from node_fdm.models.fdm_prod import FlightDynamicsModelProd
from node_fdm.models.projected_integrator import (
    ClampedEuler,
    ClampedRK4,
    _clamp_columns,
)

__all__ = [
    "ColumnStats",
    "ModelMeta",
    "NodeFDMPredictor",
]

log = structlog.get_logger("node_fdm.predictor")


class ColumnStats(BaseModel, frozen=True):
    """Per-column normalization statistics.

    Attributes:
        mean: Column mean.
        std: Column standard deviation (with epsilon offset).
        max: 99.5th percentile of absolute values.
    """

    mean: float
    std: float
    max: float
    p999: float = 0.0
    iqr: float = 0.0


class ModelMeta(BaseModel):
    """Typed schema for ``meta.json`` model metadata.

    Replaces manual ``json.load()`` + dict access with validated Pydantic
    deserialization.

    Attributes:
        architecture_name: Name of the architecture spec used.
        model_params: Tuple of ``(backbone_depth, head_depth, hidden_width)``.
        step: Integration timestep.
        shift: Window shift used during training.
        lr: Learning rate used during training.
        seq_len: Sequence length used during training.
        batch_size: Batch size used during training.
        method: ODE integration method (``"euler"`` or ``"rk4"``).
        stats_dict: Per-column normalization statistics.
        optimizer_saved: Whether an optimizer checkpoint was saved.
    """

    architecture_name: str
    architecture_spec: dict[str, Any] | None = None
    architecture_digest: str | None = None
    architecture_provider: dict[str, str | None] | None = None
    model_params: tuple[int, int, int]
    step: float
    shift: int
    lr: float
    seq_len: int
    batch_size: int
    method: str = "euler"
    stats_dict: dict[str, ColumnStats]
    optimizer_saved: bool = False
    activation: str = "silu"
    seed: int | None = None

    @classmethod
    def from_json(cls, path: Path) -> ModelMeta:
        """Load metadata from a JSON file.

        Args:
            path: Path to ``meta.json``.

        Returns:
            Validated ``ModelMeta`` instance.
        """
        text = path.read_text()
        return cls.model_validate_json(text)


class NodeFDMPredictor:
    """Predict flight trajectories using a pretrained model.

    Args:
        model_path: Directory containing ``meta.json`` and layer checkpoints.
        device: Torch device string.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)

        meta_path = self.model_path / "meta.json"
        if not meta_path.exists():
            msg = f"meta.json not found at {meta_path}"
            raise FileNotFoundError(msg)

        self.meta = ModelMeta.from_json(meta_path)
        self.spec = get(self.meta.architecture_name)
        self._validate_architecture_manifest()

        # Build stats_dict in the format expected by FlightDynamicsModelProd
        stats_plain: dict[str, dict[str, float]] = {
            col: {"mean": s.mean, "std": s.std, "max": s.max, "p999": s.p999, "iqr": s.iqr}
            for col, s in self.meta.stats_dict.items()
        }

        self.model = FlightDynamicsModelProd(
            spec=self.spec,
            stats_dict=stats_plain,
            model_params=self.meta.model_params,
            model_path=self.model_path,
            activation=self.meta.activation,
        ).to(self.device)
        self.model.eval()
        log.info(
            "predictor_loaded",
            architecture=self.meta.architecture_name,
            model_path=str(self.model_path),
        )

    def _validate_architecture_manifest(self) -> None:
        """Reject checkpoints resolved against a different provider or spec."""
        from node_fdm.architectures import architecture_digest, get_origin

        expected_digest = self.meta.architecture_digest
        current_digest = architecture_digest(self.spec)
        if expected_digest is not None and expected_digest != current_digest:
            msg = (
                f"Checkpoint architecture digest {expected_digest} does not match "
                f"installed architecture digest {current_digest}."
            )
            raise ValueError(msg)

        expected_provider = self.meta.architecture_provider
        if expected_provider is None:
            return
        current_origin = get_origin(self.meta.architecture_name)
        if current_origin is None:
            msg = "Checkpoint requires a discoverable architecture provider."
            raise ValueError(msg)
        for field in ("provider", "distribution", "version", "entry_point"):
            expected = expected_provider.get(field)
            current = getattr(current_origin, field)
            if expected is not None and expected != current:
                msg = (
                    f"Checkpoint architecture provider {field}={expected!r} does not "
                    f"match installed {field}={current!r}."
                )
                raise ValueError(msg)

    def predict_flight(
        self,
        x_init: np.ndarray,
        u_seq: np.ndarray,
        e_seq: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Generate model predictions for a flight.

        Uses the same integration pipeline as the trainer
        (``BatchNeuralODE`` + ``ClampedRK4`` / ``ClampedEuler``) so that
        control and environment inputs are linearly interpolated at
        intermediate RK4 sub-steps.

        Args:
            x_init: Initial state vector of shape ``(n_x,)``.
            u_seq: Control sequence of shape ``(n_steps, n_u)``.
            e_seq: Environment sequence of shape ``(n_steps, n_e)``.

        Returns:
            Dictionary mapping state column names to predicted arrays
            of shape ``(n_steps,)``.
        """
        if not np.isfinite(x_init).all():
            bad_mask = ~np.isfinite(x_init)
            bad_cols = [
                f"{self.spec.x_cols[i]}={x_init[i]}" for i in range(len(x_init)) if bad_mask[i]
            ]
            msg = f"x_init contains non-finite values: {', '.join(bad_cols)}"
            raise ValueError(msg)

        n_steps = u_seq.shape[0]
        step = self.meta.step

        # Resolve physical bounds from architecture spec
        x_bounds_idx = self._resolve_bounds(self.spec.x_bounds, self.spec.x_cols)
        dx_bounds_idx = self._resolve_bounds(self.spec.dx_bounds, self.spec.dx_cols)

        # Build tensors with batch dim = 1 (same shape as trainer).
        # Pad u/e with a repeated last row so that the solver can
        # interpolate at the final time point (t_grid has n_steps+1
        # entries: x0 at t=0 plus n_steps predictions).
        x0 = torch.tensor(x_init, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_pad = np.concatenate([u_seq, u_seq[-1:]], axis=0)
        e_pad = np.concatenate([e_seq, e_seq[-1:]], axis=0)
        u_t = torch.tensor(u_pad, dtype=torch.float32, device=self.device).unsqueeze(0)
        e_t = torch.tensor(e_pad, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_grid = torch.arange(
            0, (n_steps + 1) * step, step, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            self.model.reset_history()
            func = BatchNeuralODE(self.model, u_t, e_t, t_grid, dx_bounds=dx_bounds_idx)

            project_fn = (lambda x: _clamp_columns(x, x_bounds_idx)) if x_bounds_idx else None
            solver_kwargs = {"atol": 1e-6, "rtol": 1e-3, "step_size": step}

            if self.meta.method == "rk4":
                solver = ClampedRK4(func, x0, project_fn=project_fn, **solver_kwargs)
            else:
                solver = ClampedEuler(func, x0, project_fn=project_fn, **solver_kwargs)

            # (time, batch, n_x) — first entry is x0, skip it
            x_pred = solver.integrate(t_grid)[1:, 0, :].cpu().numpy()

        return {col: x_pred[:, j] for j, col in enumerate(self.spec.x_cols)}

    @staticmethod
    def _resolve_bounds(
        named_bounds: dict[str, tuple[float, float]],
        cols: list[str] | list[tuple[int, str]],
    ) -> dict[int, tuple[float, float]]:
        """Convert named bounds to column-index bounds."""
        if not named_bounds:
            return {}
        result: dict[int, tuple[float, float]] = {}
        for i, col in enumerate(cols):
            name: str = col[1] if isinstance(col, tuple) else col  # type: ignore[assignment]
            if name in named_bounds:
                result[i] = named_bounds[name]
        return result
