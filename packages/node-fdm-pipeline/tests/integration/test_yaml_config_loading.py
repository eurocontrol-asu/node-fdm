"""Tests for PipelineConfig and related configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from node_fdm_pipeline.config import (
    BadaConfig,
    ComputingConfig,
    GammaFilterConfig,
    MachFilterConfig,
    PipelineConfig,
    SelectedParamConfig,
)


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


class TestGammaFilterConfigYaml:
    """YAML-loading tests for GammaFilterConfig overrides."""

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


class TestSelectedParamConfigYaml:
    """YAML-loading tests for SelectedParamConfig overrides."""

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
        assert cfg.selected_params.mach.min_len == MachFilterConfig().min_len

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

    @pytest.mark.parametrize(
        "selected_params_block",
        ["", "selected_params: {}\n"],
        ids=["missing_section", "empty_section"],
    )
    def test_default_selected_params(self, tmp_path: Path, selected_params_block: str) -> None:
        """No or empty selected_params section → all defaults used."""
        config = tmp_path / "config.yaml"
        config.write_text(
            'paths:\n  data_dir: "/tmp/data"\ntypecodes:\n  - A320\n' + selected_params_block
        )
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params == SelectedParamConfig()
