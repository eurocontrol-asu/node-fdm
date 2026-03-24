"""Tests for PipelineConfig and related configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from node_fdm_pipeline.config import (
    BadaConfig,
    ComputingConfig,
    GammaFilterConfig,
    PathsConfig,
    PipelineConfig,
    SelectedParamConfig,
)


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
        # SelectedParamConfig defaults are populated
        assert cfg.selected_params.mach.tol == 0.0005

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


class TestGammaFilterConfig:
    """Tests for GammaFilterConfig — gamma detection filter parameters."""

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

    def test_yaml_gamma_override(self, tmp_path: Path) -> None:
        """YAML gamma overrides flow through PipelineConfig correctly."""
        config = tmp_path / "config.yaml"
        config.write_text("""\
paths:
  data_dir: "/tmp/data"
typecodes:
  - A320
selected_params:
  gamma:
    tol: 0.005
    min_len: 20
    use_alt: true
""")
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params.gamma.tol == 0.005
        assert cfg.selected_params.gamma.min_len == 20
        assert cfg.selected_params.gamma.use_alt is True
        # Non-overridden fields keep defaults
        assert cfg.selected_params.gamma.smooth_method == GammaFilterConfig().smooth_method

    def test_yaml_partial_gamma_override(self, tmp_path: Path) -> None:
        """Overriding one gamma field keeps the rest at defaults."""
        config = tmp_path / "config.yaml"
        config.write_text("""\
paths:
  data_dir: "/tmp/data"
typecodes:
  - A320
selected_params:
  gamma:
    tol: 0.01
""")
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params.gamma.tol == 0.01
        defaults = GammaFilterConfig()
        assert cfg.selected_params.gamma.min_len == defaults.min_len
        assert cfg.selected_params.gamma.use_alt == defaults.use_alt
        assert cfg.selected_params.gamma.smooth_window == defaults.smooth_window
        assert cfg.selected_params.gamma.smooth_method == defaults.smooth_method


class TestSelectedParamConfig:
    """Tests for SelectedParamConfig — externalized filter thresholds."""

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
        assert cfg.alt.min_len == 5
        assert cfg.gamma.tol == 0.002
        assert cfg.gamma.smooth_window == 5

    def test_from_yaml(self, tmp_path: Path) -> None:
        """YAML with custom mach.tol=0.01 overrides default."""
        config = tmp_path / "config.yaml"
        config.write_text("""\
paths:
  data_dir: "/tmp/data"
typecodes:
  - A320
selected_params:
  mach:
    tol: 0.01
""")
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params.mach.tol == 0.01
        # Other mach fields keep defaults
        assert cfg.selected_params.mach.min_len == 120

    def test_partial_override(self, tmp_path: Path) -> None:
        """YAML with only mach.tol set — other params keep defaults."""
        config = tmp_path / "config.yaml"
        config.write_text("""\
paths:
  data_dir: "/tmp/data"
typecodes:
  - A320
selected_params:
  mach:
    tol: 0.01
""")
        cfg = PipelineConfig.from_yaml(config)
        # mach.tol overridden
        assert cfg.selected_params.mach.tol == 0.01
        # cas, vz, alt, gamma keep all defaults
        assert cfg.selected_params.cas.tol == 0.75
        assert cfg.selected_params.vz.min_abs_value == 75
        assert cfg.selected_params.alt.min_abs_value == 25
        assert cfg.selected_params.gamma.tol == 0.002

    def test_missing_section(self, tmp_config: Path) -> None:
        """YAML with no selected_params → all defaults used, no error."""
        cfg = PipelineConfig.from_yaml(tmp_config)
        assert cfg.selected_params == SelectedParamConfig()

    def test_empty_section(self, tmp_path: Path) -> None:
        """selected_params: {} → all defaults used."""
        config = tmp_path / "config.yaml"
        config.write_text("""\
paths:
  data_dir: "/tmp/data"
typecodes:
  - A320
selected_params: {}
""")
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params == SelectedParamConfig()
