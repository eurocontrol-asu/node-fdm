"""Tests for PipelineConfig and related configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from node_fdm_pipeline.config import (
    BadaConfig,
    ComputingConfig,
    PathsConfig,
    PipelineConfig,
)


class TestPathsConfig:
    """Tests for PathsConfig model."""

    def test_resolve_models_dir(self) -> None:
        """PathsConfig.resolve() returns data_dir / sub-dir value."""
        paths = PathsConfig(data_dir=Path("/data"))
        assert paths.resolve("models_dir") == Path("/data/models")

    def test_resolve_download_dir(self) -> None:
        """PathsConfig.resolve() works for download_dir."""
        paths = PathsConfig(data_dir=Path("/data"))
        assert paths.resolve("download_dir") == Path("/data/downloaded_parquet")

    def test_resolve_invalid_name(self) -> None:
        """PathsConfig.resolve() raises for invalid attribute name."""
        paths = PathsConfig(data_dir=Path("/data"))
        with pytest.raises(AttributeError):
            paths.resolve("nonexistent")

    def test_defaults(self) -> None:
        """PathsConfig has sensible defaults for all sub-dirs."""
        paths = PathsConfig(data_dir=Path("/data"))
        assert paths.preprocess_dir == "preprocessed_parquet"
        assert paths.process_dir == "processed_flights"
        assert paths.predicted_dir == "predicted_flights"
        assert paths.bada_dir == "bada_flights"
        assert paths.figure_dir == "figures"
        assert paths.era5_cache_dir == "era5_cache"


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_from_opensky_yaml(self, opensky_config_path: Path) -> None:
        """PipelineConfig loads the real OpenSky config YAML."""
        cfg = PipelineConfig.from_yaml(opensky_config_path)
        assert len(cfg.typecodes) == 11
        assert "A320" in cfg.typecodes
        assert cfg.paths.data_dir == Path("TODO")
        assert cfg.computing.default_cpu_count == 35
        assert len(cfg.era5_features) == 3

    def test_from_qar_yaml(self, qar_config_path: Path) -> None:
        """PipelineConfig loads the real QAR config YAML with defaults."""
        cfg = PipelineConfig.from_yaml(qar_config_path)
        assert cfg.typecodes == ["A320"]
        assert cfg.era5_features == []  # Default for missing field
        assert cfg.bada == BadaConfig()  # Default for missing section
        assert cfg.computing.default_cpu_count == 35

    def test_empty_typecodes_raises(self, empty_typecodes_config: Path) -> None:
        """PipelineConfig rejects empty typecodes list."""
        with pytest.raises(ValidationError, match="typecode"):
            PipelineConfig.from_yaml(empty_typecodes_config)

    def test_minimal_config(self, tmp_config: Path) -> None:
        """PipelineConfig loads minimal valid config."""
        cfg = PipelineConfig.from_yaml(tmp_config)
        assert cfg.typecodes == ["A320"]
        assert cfg.paths.data_dir is not None
        assert cfg.computing == ComputingConfig()  # Default
        assert cfg.bada == BadaConfig()  # Default

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """PipelineConfig raises for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml(tmp_path / "missing.yaml")

    def test_frozen(self, tmp_config: Path) -> None:
        """PipelineConfig is immutable (frozen)."""
        cfg = PipelineConfig.from_yaml(tmp_config)
        with pytest.raises(ValidationError):
            cfg.typecodes = ["B738"]  # type: ignore[misc]

    def test_extra_keys_ignored(self, tmp_path: Path) -> None:
        """PipelineConfig ignores unknown keys in YAML (not extra='forbid')."""
        config = tmp_path / "config.yaml"
        config.write_text(
            """\
paths:
  data_dir: "/tmp/data"

typecodes:
  - A320

unknown_section:
  foo: bar
"""
        )
        # Should not raise — extra keys are ignored by default
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.typecodes == ["A320"]
