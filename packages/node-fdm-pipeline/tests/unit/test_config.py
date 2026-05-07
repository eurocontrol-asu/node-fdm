"""Unit tests for PipelineConfig models (pure, no I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from node_fdm_pipeline.config import GammaFilterConfig, PathsConfig, SelectedParamConfig


class TestPathsConfig:
    """Tests for PathsConfig model."""

    def test_resolve_models_dir(self) -> None:
        """PathsConfig.resolve() returns data_dir / sub-dir value."""
        paths = PathsConfig(data_dir=Path("/data"))
        assert paths.resolve("models_dir") == Path("/data/models")

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


class TestGammaFilterConfigUnit:
    """Unit tests for GammaFilterConfig — pure model behaviour."""

    def test_default_fields(self) -> None:
        """GammaFilterConfig exposes all expected fields with correct types."""
        cfg = GammaFilterConfig()
        assert isinstance(cfg.tol, float)
        assert isinstance(cfg.min_len, int)
        assert isinstance(cfg.use_alt, bool)
        assert isinstance(cfg.smooth_window, int)
        assert isinstance(cfg.smooth_method, str)

    def test_frozen(self) -> None:
        """GammaFilterConfig is immutable (frozen=True)."""
        cfg = GammaFilterConfig()
        with pytest.raises(ValidationError):
            cfg.tol = 0.5  # type: ignore[misc]

    def test_custom_values(self) -> None:
        """GammaFilterConfig accepts custom values."""
        cfg = GammaFilterConfig(tol=0.01, min_len=30, use_alt=True)
        assert cfg.tol == 0.01
        assert cfg.min_len == 30
        assert cfg.use_alt is True


class TestSelectedParamConfigUnit:
    """Unit tests for SelectedParamConfig — pure default values."""

    def test_default_values(self) -> None:
        """All v2 defaults populated correctly."""
        cfg = SelectedParamConfig()
        assert cfg.mach.tol == 0.0005
        assert cfg.mach.min_len == 120
        assert cfg.mach.alt_threshold == 15000
        assert cfg.mach.use_alt is True
        assert cfg.cas.tol == 0.75
        assert cfg.cas.smooth_method == "savgol"
        assert cfg.vz.min_abs_value == 75
        assert cfg.alt.tol == 25
        assert cfg.alt.min_len == 6
        assert cfg.gamma.tol == 0.002
        assert cfg.gamma.smooth_window == 5
