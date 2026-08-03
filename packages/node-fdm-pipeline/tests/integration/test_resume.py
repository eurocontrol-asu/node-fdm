"""Tests for the resume command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import torch
from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import ODETrainer, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta_json(
    model_dir: Path,
    *,
    architecture_name: str = "node_adsb_v1",
    method: str = "euler",
    seq_len: int = 60,
    lr: float = 1e-3,
) -> Path:
    """Write a minimal meta.json into *model_dir* and return the path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "architecture_name": architecture_name,
        "model_params": [2, 1, 48],
        "step": 1.0,
        "shift": seq_len,
        "lr": lr,
        "seq_len": seq_len,
        "batch_size": 512,
        "method": method,
        "stats_dict": {
            "col1": {"mean": 0.0, "std": 1.0, "max": 3.0},
        },
        "optimizer_saved": False,
    }
    meta_path = model_dir / "meta.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path


def _make_delta_table(path: Path, *, typecodes: list[str] | None = None) -> None:
    """Create a minimal Delta Table with required columns."""
    typecodes = typecodes or ["A320"]
    rng = np.random.default_rng(42)
    rows = []
    for acft in typecodes:
        for i in range(4):
            fid = f"abc123_{acft}_{i:02d}_s0"
            for _j in range(20):
                rows.append(
                    {
                        "meta_flight_id": fid,
                        "meta_split": ["train", "train", "val", "test"][i],
                        "meta_aircraft_type": acft,
                        "fdm_flag_valid": True,
                        "fdm_flag_distance_ok": True,
                        "raw_alt_m": float(rng.uniform(300, 10000)),
                        "era_tas_ms": float(rng.uniform(100, 250)),
                    }
                )
    df = pl.DataFrame(rows)
    df.write_delta(str(path), mode="overwrite")


def _make_config(tmp_path: Path) -> Path:
    """Create a valid YAML config and Delta Table."""
    data_dir = tmp_path / "data"
    models_dir = data_dir / "models"
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""
    )

    _make_delta_table(data_dir / "flights.delta", typecodes=["A320"])
    return config


def _make_smooth_dataset(
    n_samples: int = 16,
    seq_len: int = 5,
    n_x: int = 4,
    n_u: int = 8,
    n_e: int = 4,
) -> FlightDataset:
    """Create a dataset with smooth linear trajectories for stable ODE integration."""
    torch.manual_seed(42)
    samples: list[FlightSample] = []
    for _ in range(n_samples):
        x0 = torch.randn(n_x) * 0.1
        velocity = torch.randn(n_x) * 0.01
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


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestResumeUnit:
    """Unit tests for resume command logic."""

    def test_resume_loads_meta(self, tmp_path: Path) -> None:
        """Create model dir with meta.json → ModelMeta loaded with correct fields."""
        from node_fdm.predictor import ModelMeta

        model_dir = tmp_path / "node_adsb_v1_A320"
        _make_meta_json(
            model_dir,
            architecture_name="node_adsb_v1",
            method="rk4",
        )

        meta = ModelMeta.from_json(model_dir / "meta.json")
        assert meta.architecture_name == "node_adsb_v1"
        assert meta.method == "rk4"

    def test_resume_infers_arch_from_meta(self, tmp_path: Path) -> None:
        """Meta with architecture_name='node_adsb_v1' → correct architecture resolved."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1")
        config = _make_config(tmp_path)

        with (
            patch("node_fdm_pipeline.commands.resume.resolve_architecture") as mock_resolve,
            patch("node_fdm.loader.get_train_val_data") as mock_get_data,
            patch("node_fdm.trainer.ODETrainer") as mock_trainer_cls,
        ):
            mock_resolve.return_value = MagicMock(
                name="node_adsb_v1",
                x_cols=["a"],
                u_cols=["b"],
                e0_cols=["c"],
                dx_cols=[(1, "d")],
            )
            mock_get_data.return_value = (MagicMock(), MagicMock())
            mock_trainer_cls.return_value = MagicMock()

            run_resume(
                model=model_dir,
                config=config,
                device="cpu",
            )

            # Canonical checkpoint names are resolved directly through providers.
            mock_resolve.assert_called_once_with("node_adsb_v1")

    def test_resume_missing_meta_json(self, tmp_path: Path) -> None:
        """Point --model to empty dir → SystemExit with clear error message."""
        from node_fdm_pipeline.commands.resume import run_resume

        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir()
        config = _make_config(tmp_path)

        with pytest.raises(SystemExit, match=r"meta\.json"):
            run_resume(
                model=empty_dir,
                config=config,
                device="cpu",
            )


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestResumeFunctional:
    """Functional tests for resume training."""

    def test_resume_training_loop(self, tmp_path: Path) -> None:
        """Train 2 epochs, save, resume 2 more → 4 epochs total, new checkpoint."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_resume",
            epochs=2,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            seq_len=5,
            step=0.01,
            method="euler",
        )

        train_ds = _make_smooth_dataset(n_samples=8, seq_len=5)
        val_ds = _make_smooth_dataset(n_samples=4, seq_len=5)

        # Phase 1: train 2 epochs
        trainer1 = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=model_dir,
        )
        records1 = trainer1.train()
        assert len(records1) == 2

        # Phase 2: resume 2 more epochs
        resume_cfg = cfg.model_copy(update={"epochs": 2})
        trainer2 = ODETrainer(
            config=resume_cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=model_dir,
        )
        trainer2.load_optimizer_state()
        records2 = trainer2.train()

        assert len(records2) == 2

        # Combined: 4 epochs worth of training
        total_epochs = len(records1) + len(records2)
        assert total_epochs == 4

        # New checkpoint saved
        checkpoint_dir = model_dir / "test_resume"
        assert (checkpoint_dir / "meta.json").exists()
        assert (checkpoint_dir / "optimizer.pt").exists()

    def test_resume_with_lr_override(self, tmp_path: Path) -> None:
        """Resume with --lr 1e-4 → TrainingConfig uses overridden LR, not original."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1", lr=1e-3)
        config = _make_config(tmp_path)

        with (
            patch("node_fdm_pipeline.commands.resume.resolve_architecture") as mock_resolve,
            patch("node_fdm.loader.get_train_val_data") as mock_get_data,
            patch("node_fdm.trainer.ODETrainer") as mock_trainer_cls,
        ):
            mock_resolve.return_value = MagicMock(
                name="node_adsb_v1",
                x_cols=["a"],
                u_cols=["b"],
                e0_cols=["c"],
                dx_cols=[(1, "d")],
            )
            mock_get_data.return_value = (MagicMock(), MagicMock())
            mock_trainer_cls.return_value = MagicMock()

            run_resume(
                model=model_dir,
                config=config,
                lr=1e-4,
                device="cpu",
            )

            training_config = mock_trainer_cls.call_args.kwargs["config"]
            assert training_config.lr == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestResumeEdgeCases:
    """Edge-case tests for resume."""

    def test_no_optimizer_pt(self, tmp_path: Path) -> None:
        """Resume from checkpoint without optimizer.pt → fresh optimizer, log warning."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1")
        config = _make_config(tmp_path)

        # No optimizer.pt in model_dir

        with (
            patch("node_fdm_pipeline.commands.resume.resolve_architecture") as mock_resolve,
            patch("node_fdm.loader.get_train_val_data") as mock_get_data,
            patch("node_fdm.trainer.ODETrainer") as mock_trainer_cls,
        ):
            mock_resolve.return_value = MagicMock(
                name="node_adsb_v1",
                x_cols=["a"],
                u_cols=["b"],
                e0_cols=["c"],
                dx_cols=[(1, "d")],
            )
            mock_get_data.return_value = (MagicMock(), MagicMock())
            mock_trainer = MagicMock()
            mock_trainer_cls.return_value = mock_trainer

            run_resume(
                model=model_dir,
                config=config,
                device="cpu",
            )

            # load_optimizer_state should still be called (handles missing file internally)
            mock_trainer.load_optimizer_state.assert_called_once()

    def test_overwrite_mode(self, tmp_path: Path) -> None:
        """--overwrite flag → saves in same model dir, overwrites checkpoints."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1")
        config = _make_config(tmp_path)

        with (
            patch("node_fdm_pipeline.commands.resume.resolve_architecture") as mock_resolve,
            patch("node_fdm.loader.get_train_val_data") as mock_get_data,
            patch("node_fdm.trainer.ODETrainer") as mock_trainer_cls,
        ):
            mock_resolve.return_value = MagicMock(
                name="node_adsb_v1",
                x_cols=["a"],
                u_cols=["b"],
                e0_cols=["c"],
                dx_cols=[(1, "d")],
            )
            mock_get_data.return_value = (MagicMock(), MagicMock())
            mock_trainer_cls.return_value = MagicMock()

            run_resume(
                model=model_dir,
                config=config,
                overwrite=True,
                device="cpu",
            )

            # model_dir passed to ODETrainer should be the parent (models/)
            # so that model_name recreates the same subdir
            call_kwargs = mock_trainer_cls.call_args.kwargs
            assert call_kwargs["model_dir"] == model_dir.parent

    def test_different_seq_len_on_resume(self, tmp_path: Path) -> None:
        """--seq-len 120 (was 60) → new datasets built with new seq-len."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1", seq_len=60)
        config = _make_config(tmp_path)

        with (
            patch("node_fdm_pipeline.commands.resume.resolve_architecture") as mock_resolve,
            patch("node_fdm.loader.get_train_val_data") as mock_get_data,
            patch("node_fdm.trainer.ODETrainer") as mock_trainer_cls,
        ):
            mock_resolve.return_value = MagicMock(
                name="node_adsb_v1",
                x_cols=["a"],
                u_cols=["b"],
                e0_cols=["c"],
                dx_cols=[(1, "d")],
            )
            mock_get_data.return_value = (MagicMock(), MagicMock())
            mock_trainer_cls.return_value = MagicMock()

            run_resume(
                model=model_dir,
                config=config,
                seq_len=120,
                device="cpu",
            )

            # TrainingConfig should use new seq_len
            training_config = mock_trainer_cls.call_args.kwargs["config"]
            assert training_config.seq_len == 120

            # get_train_val_data should be called with new seq_len
            data_kwargs = mock_get_data.call_args.kwargs
            assert data_kwargs["seq_len"] == 120

    def test_resume_e1_cols_loaded(self, tmp_path: Path) -> None:
        """E1 columns forwarded to get_train_val_data on resume.

        When e1_cols are passed, FlightSample.e1 tensors are populated,
        ensuring resumed training uses the same feature set as initial training.
        """
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1")
        config = _make_config(tmp_path)

        with (
            patch("node_fdm_pipeline.commands.resume.resolve_architecture") as mock_resolve,
            patch("node_fdm.loader.get_train_val_data") as mock_get_data,
            patch("node_fdm.trainer.ODETrainer") as mock_trainer_cls,
        ):
            mock_resolve.return_value = MagicMock(
                name="node_adsb_v1",
                x_cols=["a"],
                u_cols=["b"],
                e0_cols=["c"],
                e1_cols=["fdm_tas_diff_ms", "fdm_alt_diff_m"],
                dx_cols=[(1, "d")],
            )
            mock_get_data.return_value = (MagicMock(), MagicMock())
            mock_trainer_cls.return_value = MagicMock()

            run_resume(
                model=model_dir,
                config=config,
                device="cpu",
            )

            # get_train_val_data must receive e1_cols kwarg
            data_kwargs = mock_get_data.call_args.kwargs
            assert "e1_cols" in data_kwargs, "e1_cols not passed to get_train_val_data"
            assert data_kwargs["e1_cols"] == ["fdm_tas_diff_ms", "fdm_alt_diff_m"]

    def test_resume_unknown_architecture(self, tmp_path: Path) -> None:
        """Meta with unknown architecture_name → SystemExit with known archs listed."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "bogus_arch_A320"
        _make_meta_json(model_dir, architecture_name="bogus_arch")
        config = _make_config(tmp_path)

        with pytest.raises(SystemExit, match="Unknown architecture_name"):
            run_resume(model=model_dir, config=config, device="cpu")

    def test_resume_missing_delta_table(self, tmp_path: Path) -> None:
        """Valid meta but missing delta table → SystemExit."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1")

        # Config pointing to non-existent delta table
        data_dir = tmp_path / "data_no_delta"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""
        )

        with (
            pytest.raises(SystemExit, match="Delta table not found"),
        ):
            run_resume(model=model_dir, config=config, device="cpu")

    def test_resume_empty_dataset_for_typecode(self, tmp_path: Path) -> None:
        """No data rows for inferred typecode → SystemExit."""
        from node_fdm_pipeline.commands.resume import run_resume

        # Model dir implies typecode "B777" but delta only has A320
        model_dir = tmp_path / "models" / "node_adsb_v1_B777"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1")
        config = _make_config(tmp_path)  # creates delta with A320 only

        with (
            pytest.raises(SystemExit, match="No data for typecode"),
        ):
            run_resume(model=model_dir, config=config, device="cpu")

    def test_resume_e1_cols_has_gamma_diff(self, tmp_path: Path) -> None:
        """Resuming adsb model passes fdm_gamma_diff_rad in e1_cols to loader."""
        from node_fdm_pipeline.commands.resume import run_resume

        model_dir = tmp_path / "models" / "node_adsb_v1_A320"
        _make_meta_json(model_dir, architecture_name="node_adsb_v1")
        config = _make_config(tmp_path)

        with (
            patch("node_fdm.loader.get_train_val_data") as mock_get_data,
            patch("node_fdm.trainer.ODETrainer") as mock_trainer_cls,
        ):
            mock_get_data.return_value = (MagicMock(), MagicMock())
            mock_trainer_cls.return_value = MagicMock()

            run_resume(
                model=model_dir,
                config=config,
                device="cpu",
            )

            data_kwargs = mock_get_data.call_args.kwargs
            assert "e1_cols" in data_kwargs, "e1_cols not passed to get_train_val_data"
            assert "fdm_gamma_diff_rad" in data_kwargs["e1_cols"]
