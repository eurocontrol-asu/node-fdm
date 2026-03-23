"""Prediction helper with typed model metadata.

Replaces the legacy ``NodeFDMPredictor`` that used ``dict[str, Any]``
column definitions and pandas DataFrames with Pydantic-typed
:class:`ModelMeta` and NumPy array outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
from pydantic import BaseModel

from node_fdm.architectures.registry import get
from node_fdm.models.fdm_prod import FlightDynamicsModelProd

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
    """

    architecture_name: str
    model_params: tuple[int, int, int]
    step: float
    shift: int
    lr: float
    seq_len: int
    batch_size: int
    method: str = "euler"
    stats_dict: dict[str, ColumnStats]

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

        # Build stats_dict in the format expected by FlightDynamicsModelProd
        stats_plain: dict[str, dict[str, float]] = {
            col: {"mean": s.mean, "std": s.std, "max": s.max}
            for col, s in self.meta.stats_dict.items()
        }

        self.model = FlightDynamicsModelProd(
            spec=self.spec,
            stats_dict=stats_plain,
            model_params=self.meta.model_params,
            model_path=self.model_path,
        ).to(self.device)
        self.model.eval()
        log.info(
            "predictor_loaded",
            architecture=self.meta.architecture_name,
            model_path=str(self.model_path),
        )

    def predict_flight(
        self,
        x_init: np.ndarray,
        u_seq: np.ndarray,
        e_seq: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Generate model predictions for a flight.

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
        results: dict[str, list[float]] = {col: [] for col in self.spec.x_cols}

        x_t = torch.tensor(x_init, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            for i in range(n_steps):
                u_t = torch.tensor(
                    u_seq[i],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                e_t = torch.tensor(
                    e_seq[i],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                if self.meta.method == "rk4":
                    x_t = self._rk4_step(x_t, u_t, e_t)
                else:
                    x_t = self._euler_step(x_t, u_t, e_t)

                for j, col in enumerate(self.spec.x_cols):
                    results[col].append(x_t[0, j].item())

        return {col: np.array(vals) for col, vals in results.items()}

    def _euler_step(
        self,
        x_t: torch.Tensor,
        u_t: torch.Tensor,
        e_t: torch.Tensor,
    ) -> torch.Tensor:
        """Advance state by one Euler step."""
        self.model.reset_history()
        dx = self.model(x_t, u_t, e_t)
        x_next = x_t.clone()
        for j, (coeff, _col) in enumerate(self.spec.dx_cols):
            x_next[0, j] = x_t[0, j] + coeff * self.meta.step * dx[0, j]
        return x_next

    def _rk4_step(
        self,
        x_t: torch.Tensor,
        u_t: torch.Tensor,
        e_t: torch.Tensor,
    ) -> torch.Tensor:
        """Advance state by one classical RK4 step."""
        dt = self.meta.step

        self.model.reset_history()
        k1 = self.model(x_t, u_t, e_t)

        x2 = x_t.clone()
        for j, (coeff, _col) in enumerate(self.spec.dx_cols):
            x2[0, j] = x_t[0, j] + 0.5 * coeff * dt * k1[0, j]
        self.model.reset_history()
        k2 = self.model(x2, u_t, e_t)

        x3 = x_t.clone()
        for j, (coeff, _col) in enumerate(self.spec.dx_cols):
            x3[0, j] = x_t[0, j] + 0.5 * coeff * dt * k2[0, j]
        self.model.reset_history()
        k3 = self.model(x3, u_t, e_t)

        x4 = x_t.clone()
        for j, (coeff, _col) in enumerate(self.spec.dx_cols):
            x4[0, j] = x_t[0, j] + coeff * dt * k3[0, j]
        self.model.reset_history()
        k4 = self.model(x4, u_t, e_t)

        x_next = x_t.clone()
        for j, (coeff, _col) in enumerate(self.spec.dx_cols):
            x_next[0, j] = (
                x_t[0, j] + coeff * dt * (k1[0, j] + 2 * k2[0, j] + 2 * k3[0, j] + k4[0, j]) / 6
            )
        return x_next
