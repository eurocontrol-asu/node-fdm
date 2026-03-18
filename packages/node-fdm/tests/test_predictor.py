"""Tests for ModelMeta, ColumnStats, and NodeFDMPredictor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from node_fdm.predictor import ColumnStats, ModelMeta, NodeFDMPredictor


class TestColumnStats:
    """Unit tests for ColumnStats Pydantic model."""

    def test_roundtrip(self) -> None:
        """model_dump_json → model_validate_json preserves fields."""
        cs = ColumnStats(mean=1.5, std=0.3, max=5.0)
        restored = ColumnStats.model_validate_json(cs.model_dump_json())
        assert restored == cs

    def test_frozen(self) -> None:
        """ColumnStats is immutable."""
        cs = ColumnStats(mean=1.0, std=0.5, max=3.0)
        with pytest.raises(Exception):  # noqa: B017
            cs.mean = 2.0  # type: ignore[misc]


class TestModelMeta:
    """Unit tests for ModelMeta Pydantic model."""

    def test_roundtrip(self) -> None:
        """Full roundtrip preserves all fields."""
        meta = ModelMeta(
            architecture_name="opensky_2025",
            model_params=(2, 1, 48),
            step=1.0,
            shift=60,
            lr=1e-3,
            seq_len=60,
            batch_size=512,
            stats_dict={
                "altitude_m": ColumnStats(mean=5000.0, std=3000.0, max=12000.0),
                "tas_ms": ColumnStats(mean=200.0, std=50.0, max=280.0),
            },
        )
        json_str = meta.model_dump_json()
        restored = ModelMeta.model_validate_json(json_str)
        assert restored.architecture_name == "opensky_2025"
        assert restored.model_params == (2, 1, 48)
        assert len(restored.stats_dict) == 2
        assert restored.stats_dict["altitude_m"].mean == pytest.approx(5000.0)

    def test_from_json_file(self, tmp_path: Path) -> None:
        """Load from a JSON file on disk."""
        meta_data = {
            "architecture_name": "opensky_2025",
            "model_params": [2, 1, 48],
            "step": 1.0,
            "shift": 60,
            "lr": 0.001,
            "seq_len": 60,
            "batch_size": 512,
            "stats_dict": {
                "col1": {"mean": 0.0, "std": 1.0, "max": 3.0},
            },
        }
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps(meta_data))

        meta = ModelMeta.from_json(meta_path)
        assert meta.architecture_name == "opensky_2025"
        assert "col1" in meta.stats_dict


class TestNodeFDMPredictor:
    """Functional tests for NodeFDMPredictor."""

    def test_missing_meta_raises(self, tmp_path: Path) -> None:
        """Missing meta.json raises FileNotFoundError."""
        from node_fdm.predictor import NodeFDMPredictor

        with pytest.raises(FileNotFoundError, match=r"meta\.json"):
            NodeFDMPredictor(model_path=tmp_path)

    def test_predict_flight_shapes(self, tmp_path: Path) -> None:
        """predict_flight returns arrays with correct shapes."""
        from node_fdm.architectures.registry import get
        from node_fdm.dataset import FlightSample, compute_stats
        from node_fdm.models.fdm import FlightDynamicsModel
        from node_fdm.predictor import NodeFDMPredictor

        spec = get("opensky_2025")
        n_x = len(spec.x_cols)
        n_u = len(spec.u_cols)
        n_e = len(spec.e0_cols)

        # Build dummy stats
        samples = [
            FlightSample(
                x=torch.randn(10, n_x),
                u=torch.randn(10, n_u),
                e=torch.randn(10, n_e),
                dx=torch.randn(10, n_x),
            )
        ]
        dx_col_names = [col for _, col in spec.dx_cols]
        stats = compute_stats(
            samples,
            spec.x_cols,
            spec.u_cols,
            spec.e0_cols,
            dx_col_names,
        )

        # Create model and save checkpoint + meta
        model = FlightDynamicsModel(spec, stats)
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Save layer checkpoints
        for layer_spec in spec.layers:
            if layer_spec.trainable:
                layer = model.layers_dict[layer_spec.name]
                save_dict = {
                    "layer_state": layer.state_dict(),
                    "best_val_loss": 0.1,
                    "epoch": 1,
                }
                torch.save(save_dict, model_dir / f"{layer_spec.name}.pt")

        # Save meta.json
        meta_data = {
            "architecture_name": "opensky_2025",
            "model_params": [2, 1, 48],
            "step": 1.0,
            "shift": 60,
            "lr": 0.001,
            "seq_len": 60,
            "batch_size": 512,
            "stats_dict": stats,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta_data))

        # Run predictor
        predictor = NodeFDMPredictor(model_path=model_dir, device="cpu")

        n_steps = 5
        x_init = np.random.randn(n_x).astype(np.float32)
        u_seq = np.random.randn(n_steps, n_u).astype(np.float32)
        e_seq = np.random.randn(n_steps, n_e).astype(np.float32)

        result = predictor.predict_flight(x_init, u_seq, e_seq)

        assert isinstance(result, dict)
        assert len(result) == n_x
        for col in spec.x_cols:
            assert col in result
            assert result[col].shape == (n_steps,)
            assert np.isfinite(result[col]).all()

    def _make_predictor(self, tmp_path: Path) -> NodeFDMPredictor:
        """Create a minimal predictor for guard tests."""
        from node_fdm.architectures.registry import get
        from node_fdm.dataset import FlightSample, compute_stats
        from node_fdm.models.fdm import FlightDynamicsModel
        from node_fdm.predictor import NodeFDMPredictor

        spec = get("opensky_2025")
        n_x = len(spec.x_cols)
        n_u = len(spec.u_cols)
        n_e = len(spec.e0_cols)

        samples = [
            FlightSample(
                x=torch.randn(10, n_x),
                u=torch.randn(10, n_u),
                e=torch.randn(10, n_e),
                dx=torch.randn(10, n_x),
            )
        ]
        dx_col_names = [col for _, col in spec.dx_cols]
        stats = compute_stats(samples, spec.x_cols, spec.u_cols, spec.e0_cols, dx_col_names)

        model = FlightDynamicsModel(spec, stats)
        model_dir = tmp_path / "guard_model"
        model_dir.mkdir()
        for layer_spec in spec.layers:
            if layer_spec.trainable:
                layer = model.layers_dict[layer_spec.name]
                save_dict = {"layer_state": layer.state_dict(), "best_val_loss": 0.1, "epoch": 1}
                torch.save(save_dict, model_dir / f"{layer_spec.name}.pt")
        meta_data = {
            "architecture_name": "opensky_2025",
            "model_params": [2, 1, 48],
            "step": 1.0,
            "shift": 60,
            "lr": 0.001,
            "seq_len": 60,
            "batch_size": 512,
            "stats_dict": stats,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta_data))
        return NodeFDMPredictor(model_path=model_dir, device="cpu")

    def test_predict_flight_raises_on_nan_x_init(self, tmp_path: Path) -> None:
        """ValueError raised when x_init contains NaN, message includes column name."""
        predictor = self._make_predictor(tmp_path)
        n_x = len(predictor.spec.x_cols)
        n_u = len(predictor.spec.u_cols)
        n_e = len(predictor.spec.e0_cols)

        x_init = np.zeros(n_x, dtype=np.float32)
        x_init[1] = float("nan")  # altitude_m

        with pytest.raises(ValueError, match="altitude_m"):
            predictor.predict_flight(
                x_init,
                np.zeros((5, n_u), dtype=np.float32),
                np.zeros((5, n_e), dtype=np.float32),
            )

    def test_predict_flight_raises_on_inf_x_init(self, tmp_path: Path) -> None:
        """ValueError raised when x_init contains Inf."""
        predictor = self._make_predictor(tmp_path)
        n_x = len(predictor.spec.x_cols)
        n_u = len(predictor.spec.u_cols)
        n_e = len(predictor.spec.e0_cols)

        x_init = np.zeros(n_x, dtype=np.float32)
        x_init[1] = float("inf")

        with pytest.raises(ValueError, match="non-finite"):
            predictor.predict_flight(
                x_init,
                np.zeros((5, n_u), dtype=np.float32),
                np.zeros((5, n_e), dtype=np.float32),
            )

    def test_predict_flight_clean_x_init_ok(self, tmp_path: Path) -> None:
        """Clean x_init does not raise — prediction proceeds normally."""
        predictor = self._make_predictor(tmp_path)
        n_x = len(predictor.spec.x_cols)
        n_u = len(predictor.spec.u_cols)
        n_e = len(predictor.spec.e0_cols)

        x_init = np.array([0.0, 1000.0, 0.05, 230.0], dtype=np.float32)[:n_x]

        result = predictor.predict_flight(
            x_init,
            np.random.randn(5, n_u).astype(np.float32),
            np.random.randn(5, n_e).astype(np.float32),
        )
        assert len(result) == n_x
