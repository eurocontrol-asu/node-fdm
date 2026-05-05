"""Tests for projected integrator wiring into ODETrainer (AXM-821).

TDD tests — expected to FAIL until implementation is complete.
Covers: projected ODE loss, backward pass, soft-clamp dx, no-bounds
compatibility, grad_clip_norm default, convergence with bounds,
method dispatch, and edge cases.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import TrainingConfig, _collate_flight_samples

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ADSB_X_COLS = ["fdm_gamma_rad", "era_tas_ms"]
_ADSB_U_COLS = ["fdm_alt_target_m"]
_ADSB_E0_COLS = ["era_mach"]
_ADSB_E1_COLS: list[str] = []
_ADSB_DX_COLS: list[tuple[int, str]] = [
    (1, "fdm_d_gamma_rads"),
    (1, "fdm_d_tas_ms2"),
]

_ADSB_SPEC = ArchitectureSpec(
    name="test_adsb_bounded",
    x_cols=_ADSB_X_COLS,
    u_cols=_ADSB_U_COLS,
    e0_cols=_ADSB_E0_COLS,
    e1_cols=_ADSB_E1_COLS,
    dx_cols=_ADSB_DX_COLS,
    layers=[
        LayerSpec(
            name="data_ode",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=_ADSB_X_COLS + _ADSB_U_COLS + _ADSB_E0_COLS,
            output_cols=["fdm_d_gamma_rads", "fdm_d_tas_ms2"],
            trainable=True,
        ),
    ],
    x_bounds={
        "fdm_gamma_rad": (-0.3, 0.3),
        "era_tas_ms": (50.0, 350.0),
    },
    dx_bounds={
        "fdm_d_gamma_rads": (-0.01, 0.01),
        "fdm_d_tas_ms2": (-5.0, 5.0),
    },
)


@pytest.fixture(autouse=True)
def _register_test_spec() -> None:
    """Register the bounded test architecture spec."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        register(_ADSB_SPEC)


def _make_smooth_dataset(
    n_samples: int = 16,
    seq_len: int = 5,
    n_x: int = 2,
    n_u: int = 1,
    n_e: int = 1,
) -> FlightDataset:
    """Smooth linear trajectories within physical bounds."""
    torch.manual_seed(42)
    samples: list[FlightSample] = []
    for _ in range(n_samples):
        # Start within bounds: gamma~0, tas~200
        x0 = torch.tensor([0.0, 200.0])
        velocity = torch.tensor([0.001, 0.1])
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        x = x0.unsqueeze(0) + t * velocity.unsqueeze(0)
        dx = velocity.unsqueeze(0).expand(seq_len, n_x)
        samples.append(
            FlightSample(
                x=x,
                u=torch.randn(seq_len, n_u) * 0.01,
                e=torch.randn(seq_len, n_e) * 0.01,
                dx=dx,
            )
        )
    return FlightDataset(samples)


def _make_bounded_config(**overrides: object) -> TrainingConfig:
    """Config using the bounded test architecture."""
    defaults = {
        "architecture_name": "test_adsb_bounded",
        "model_name": "test_projected",
        "epochs": 1,
        "batch_size": 4,
        "num_workers": 0,
        "val_batch_size": 4,
        "seq_len": 5,
        "step": 0.01,
        "method": "rk4",
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestComputeBatchLossProjected:
    """Loss computation with projected integrator (bounded spec)."""

    def test_compute_batch_loss_projected(self, tmp_path: Path) -> None:
        """Synthetic batch with adsb spec + bounds → loss is finite, no NaN."""
        from node_fdm.trainer import ODETrainer

        cfg = _make_bounded_config()
        train_ds = _make_smooth_dataset(n_samples=8)
        val_ds = _make_smooth_dataset(n_samples=4)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        # Build a single batch
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, collate_fn=_collate_flight_samples
        )
        batch = next(iter(loader))

        loss = trainer._compute_batch_loss(batch)
        assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"

    def test_compute_batch_loss_backward(self, tmp_path: Path) -> None:
        """Loss .backward() produces finite gradients on all parameters."""
        from node_fdm.trainer import ODETrainer

        cfg = _make_bounded_config()
        train_ds = _make_smooth_dataset(n_samples=8)
        val_ds = _make_smooth_dataset(n_samples=4)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, collate_fn=_collate_flight_samples
        )
        batch = next(iter(loader))

        loss = trainer._compute_batch_loss(batch)
        loss.backward()  # type: ignore[no-untyped-call]

        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"


class TestBatchNeuralODESoftClamp:
    """Soft clamping of dx output in BatchNeuralODE."""

    def test_batch_neural_ode_soft_clamp(self) -> None:
        """BatchNeuralODE with dx_bounds + extreme dx → output within soft bounds."""
        from node_fdm.models.batch_neural_ode import BatchNeuralODE

        # Build a mock model that returns extreme derivatives
        class ExtremeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)

            def reset_history(self) -> None:
                pass

            def forward(self, x: torch.Tensor, _u: torch.Tensor, _e: torch.Tensor) -> torch.Tensor:
                batch = x.shape[0]
                return torch.tensor([[1.0, 50.0]] * batch)

        model = ExtremeModel()
        batch_size, seq_len, n_u, n_e = 2, 5, 1, 1
        u_seq = torch.zeros(batch_size, seq_len, n_u)
        e_seq = torch.zeros(batch_size, seq_len, n_e)
        t_grid = torch.arange(seq_len, dtype=torch.float32)

        # dx_bounds as column-index mapping: col 0 → gamma, col 1 → tas
        dx_bounds = {
            0: (-0.01, 0.01),
            1: (-5.0, 5.0),
        }

        func = BatchNeuralODE(model, u_seq, e_seq, t_grid, dx_bounds=dx_bounds)

        x = torch.zeros(batch_size, 2)
        t = torch.tensor(0.0)
        dx = func(t, x)

        # Soft clamp should keep values within bounds (with small epsilon)
        for col_idx, (lo, hi) in dx_bounds.items():
            col = dx[:, col_idx]
            assert (col >= lo - 1e-6).all(), f"dx col {col_idx} below lower bound: {col}"
            assert (col <= hi + 1e-6).all(), f"dx col {col_idx} above upper bound: {col}"


class TestTrainerNoBoundsCompat:
    """Backward compatibility: no-bounds spec still works identically."""

    def test_trainer_no_bounds_compat(self, tmp_path: Path) -> None:
        """node_adsb_v1 spec (no bounds) → training runs identically to before."""
        from node_fdm.trainer import ODETrainer

        # node_adsb_v1 has 4 x_cols, 4+ u_cols, 4+ e_cols
        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_compat",
            epochs=1,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            seq_len=5,
            step=0.01,
            method="euler",
        )

        torch.manual_seed(0)
        n_x = 4
        n_u = 8
        n_e = 4
        samples = [
            FlightSample(
                x=torch.randn(5, n_x) * 0.1,
                u=torch.randn(5, n_u) * 0.01,
                e=torch.randn(5, n_e) * 0.01,
                dx=torch.randn(5, n_x) * 0.01,
            )
            for _ in range(12)
        ]
        ds = FlightDataset(samples)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=ds,
            val_dataset=FlightDataset(samples[:4]),
            model_dir=tmp_path,
        )

        records = trainer.train()
        assert len(records) == 1
        assert records[0]["train_loss"] == records[0]["train_loss"]  # not NaN


class TestGradClipNormDefault:
    """TrainingConfig.grad_clip_norm default value."""

    def test_grad_clip_norm_default(self) -> None:
        """Default grad_clip_norm should be 10.0."""
        cfg = TrainingConfig(architecture_name="node_adsb_v1", model_name="test")
        assert cfg.grad_clip_norm == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestTrainingConvergesAdsb:
    """Convergence test with bounded adsb-like data."""

    def test_training_converges_adsb(self, tmp_path: Path) -> None:
        """5 epochs on synthetic adsb-like data with bounds → val_loss decreases, no NaN."""
        from node_fdm.trainer import ODETrainer

        cfg = _make_bounded_config(
            epochs=5,
            batch_size=4,
            lr=1e-3,
            step=0.1,
        )

        train_ds = _make_smooth_dataset(n_samples=16)
        val_ds = _make_smooth_dataset(n_samples=4)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        records = trainer.train()
        assert len(records) == 5

        # No NaN in any epoch
        for r in records:
            assert r["train_loss"] == r["train_loss"], "NaN in train_loss"
            assert r["val_loss"] == r["val_loss"], "NaN in val_loss"

        # val_loss should decrease
        assert records[-1]["val_loss"] < records[0]["val_loss"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions for projected integration."""

    def test_all_states_clamp_every_step(self, tmp_path: Path) -> None:
        """Random model + tight bounds → loss finite, gradients finite."""
        from node_fdm.trainer import ODETrainer

        # Create spec with very tight bounds that force clamping every step
        tight_spec = ArchitectureSpec(
            name="test_tight_bounds",
            x_cols=_ADSB_X_COLS,
            u_cols=_ADSB_U_COLS,
            e0_cols=_ADSB_E0_COLS,
            e1_cols=_ADSB_E1_COLS,
            dx_cols=_ADSB_DX_COLS,
            layers=_ADSB_SPEC.layers,
            x_bounds={
                "fdm_gamma_rad": (-0.001, 0.001),
                "era_tas_ms": (199.0, 201.0),
            },
            dx_bounds={
                "fdm_d_gamma_rads": (-0.0001, 0.0001),
                "fdm_d_tas_ms2": (-0.01, 0.01),
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            register(tight_spec)

        cfg = TrainingConfig(
            architecture_name="test_tight_bounds",
            model_name="test_tight",
            epochs=1,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            seq_len=5,
            step=0.01,
        )

        train_ds = _make_smooth_dataset(n_samples=8)
        val_ds = _make_smooth_dataset(n_samples=4)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, collate_fn=_collate_flight_samples
        )
        batch = next(iter(loader))
        loss = trainer._compute_batch_loss(batch)
        assert torch.isfinite(loss).item(), f"Loss not finite: {loss.item()}"

        loss.backward()  # type: ignore[no-untyped-call]
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"

    def test_method_euler_uses_clamped_euler(self, tmp_path: Path) -> None:
        """Config with method='euler' + bounds → uses ClampedEuler."""
        from unittest.mock import patch

        from node_fdm.models.projected_integrator import ClampedEuler
        from node_fdm.trainer import ODETrainer

        cfg = _make_bounded_config(method="euler")
        train_ds = _make_smooth_dataset(n_samples=8)
        val_ds = _make_smooth_dataset(n_samples=4)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        # Patch ClampedEuler to track instantiation
        original_init = ClampedEuler.__init__
        called = {"count": 0}

        def tracking_init(self: Any, *args: Any, **kwargs: Any) -> None:
            called["count"] += 1
            return original_init(self, *args, **kwargs)

        with patch.object(ClampedEuler, "__init__", tracking_init):
            loader = torch.utils.data.DataLoader(
                train_ds, batch_size=4, collate_fn=_collate_flight_samples
            )
            batch = next(iter(loader))
            trainer._compute_batch_loss(batch)

        assert called["count"] > 0, "ClampedEuler was not used for method='euler' with bounds"

    def test_method_rk4_uses_clamped_rk4(self, tmp_path: Path) -> None:
        """Config with method='rk4' + bounds → uses ClampedRK4."""
        from unittest.mock import patch

        from node_fdm.models.projected_integrator import ClampedRK4
        from node_fdm.trainer import ODETrainer

        cfg = _make_bounded_config(method="rk4")
        train_ds = _make_smooth_dataset(n_samples=8)
        val_ds = _make_smooth_dataset(n_samples=4)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        original_init = ClampedRK4.__init__
        called = {"count": 0}

        def tracking_init(self: Any, *args: Any, **kwargs: Any) -> None:
            called["count"] += 1
            return original_init(self, *args, **kwargs)

        with patch.object(ClampedRK4, "__init__", tracking_init):
            loader = torch.utils.data.DataLoader(
                train_ds, batch_size=4, collate_fn=_collate_flight_samples
            )
            batch = next(iter(loader))
            trainer._compute_batch_loss(batch)

        assert called["count"] > 0, "ClampedRK4 was not used for method='rk4' with bounds"

    def test_unsupported_method_with_bounds(self, tmp_path: Path) -> None:
        """method='dopri5' + x_bounds → fallback with warning or raises."""
        from node_fdm.trainer import ODETrainer

        cfg = _make_bounded_config(method="dopri5")
        train_ds = _make_smooth_dataset(n_samples=8)
        val_ds = _make_smooth_dataset(n_samples=4)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, collate_fn=_collate_flight_samples
        )
        batch = next(iter(loader))

        # Should either raise or emit a warning and fall back to standard odeint
        with pytest.raises((ValueError, UserWarning)):
            trainer._compute_batch_loss(batch)
