"""Tests for the ``smooth`` pipeline command."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestEKFSmoothConfig:
    """EKFSmoothConfig defaults and integration."""

    def test_default_config(self) -> None:
        """Default EKFSmoothConfig has expected values."""
        from node_fdm_pipeline.config import EKFSmoothConfig

        cfg = EKFSmoothConfig()
        assert cfg.reject_sigma == 3.0
        assert cfg.rolling_window == 17
        assert cfg.enabled is True

    def test_pipeline_config_includes_ekf(self, tmp_path: Path) -> None:
        """PipelineConfig exposes ekf_smooth field."""
        import yaml

        from node_fdm_pipeline.config import PipelineConfig

        config_data = {
            "paths": {"data_dir": str(tmp_path)},
            "typecodes": ["A320"],
            "ekf_smooth": {
                "reject_sigma": 2.5,
                "rolling_window": 21,
                "enabled": False,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        cfg = PipelineConfig.from_yaml(config_file)
        assert cfg.ekf_smooth.reject_sigma == 2.5
        assert cfg.ekf_smooth.rolling_window == 21
        assert cfg.ekf_smooth.enabled is False


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestSmoothDryRun:
    """Dry run should produce no side effects."""

    def test_dry_run_no_io(self, tmp_path: Path) -> None:
        """Smooth dry run validates config but does not touch the table."""
        import yaml

        from node_fdm_pipeline.commands.data import smooth

        config_data = {
            "paths": {"data_dir": str(tmp_path)},
            "typecodes": ["A320"],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Should not raise
        smooth(config=config_file, dry_run=True)

        # No delta table should be created
        assert not (tmp_path / "flights.delta").exists()


# ---------------------------------------------------------------------------
# Disabled config
# ---------------------------------------------------------------------------


class TestSmoothDisabled:
    """When enabled=False, smooth should skip processing."""

    def test_disabled_skips(self, tmp_path: Path) -> None:
        """Smooth with enabled=False does not process data."""
        import yaml

        from node_fdm_pipeline.commands.data import smooth

        config_data = {
            "paths": {"data_dir": str(tmp_path)},
            "typecodes": ["A320"],
            "ekf_smooth": {"enabled": False},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Should not raise even without a delta table
        smooth(config=config_file, dry_run=False)
