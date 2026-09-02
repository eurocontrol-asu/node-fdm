"""Tests for PipelineConfig and related configuration models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from _config_fixtures import selected_param_config
from node_fdm_pipeline.config import (
    BadaConfig,
    ComputingConfig,
    PipelineConfig,
)


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    @pytest.mark.integration
    def test_fleet_run_section_is_loaded(
        self,
        tmp_path: Path,
        make_config: Callable[..., Path],
    ) -> None:
        """AC1: YAML fleet_run values are exposed by PipelineConfig."""
        lease_path = (tmp_path / "shared" / "trino.lease").resolve()
        config = make_config(
            tmp_path / "config.yaml",
            "/tmp/data",
            extra=(
                "fleet_run:\n"
                f'  lease_path: "{lease_path}"\n'
                "  lease_ttl_s: 120\n"
                "  disk_min_gib: 8.5\n"
            ),
        )

        cfg = PipelineConfig.from_yaml(config)

        assert cfg.fleet_run is not None
        assert cfg.fleet_run.lease_path == lease_path
        assert cfg.fleet_run.lease_ttl_s == 120
        assert cfg.fleet_run.disk_min_gib == 8.5

    def test_from_opensky_yaml(self, opensky_config_path: Path) -> None:
        """PipelineConfig loads the real OpenSky config YAML."""
        cfg = PipelineConfig.from_yaml(opensky_config_path)
        assert len(cfg.typecodes) == 11
        assert "A320" in cfg.typecodes
        assert cfg.paths.data_dir == Path("TODO")
        assert cfg.computing.default_cpu_count == 35
        assert len(cfg.era5_features) == 3
        # Legacy savgol fields keep their defaults when the YAML is silent.
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

    def test_extra_keys_ignored(self, tmp_path: Path, make_config: Callable[..., Path]) -> None:
        """PipelineConfig ignores unknown keys in YAML (not extra='forbid')."""
        config = make_config(
            tmp_path / "config.yaml",
            "/tmp/data",
            extra="unknown_section:\n  foo: bar\n",
        )
        # Should not raise — extra keys are ignored by default
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.typecodes == ["A320"]


class TestGammaFilterConfigYaml:
    """YAML-loading tests for GammaFilterConfig overrides."""

    def test_yaml_gamma_override(self, tmp_path: Path) -> None:
        """YAML gamma overrides flow through PipelineConfig correctly."""
        config = tmp_path / "config.yaml"
        config.write_text(
            'paths:\n  data_dir: "/tmp/data"\ntypecodes:\n  - A320\n'
            + """\
selected_params:
  mach:
    {sigma_s: 8.0, sigma_r: 0.01, n_passes: 2, slope_tol: 3.0e-4,
     flat_tol: 5.0e-2, min_len: 15}
  cas:
    {cutoff_s: 180.0, sigma_s: 8.0, sigma_r: 3.0, n_passes: 2, slope_tol: 0.09,
     flat_tol: 20.0, min_len: 5}
  vz: {sigma_s: 6.0, sigma_r: 100.0, slope_tol: 50.0, flat_tol: 100.0, min_len: 10}
  alt: {sigma_s: 6.0, sigma_r: 20.0, n_passes: 2, tol_ftmin: 150.0, min_len: 6}
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002
    slope_tol: 1.2e-3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    tol: 0.005
    min_len: 20
    use_alt: true
"""
        )
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params.gamma.tol == 0.005
        assert cfg.selected_params.gamma.min_len == 20
        assert cfg.selected_params.gamma.use_alt is True
        # Non-overridden legacy fields keep defaults
        assert cfg.selected_params.gamma.smooth_method == "savgol"

    def test_yaml_partial_gamma_override(self, tmp_path: Path) -> None:
        """Overriding one legacy gamma field keeps the other legacy fields at defaults."""
        config = tmp_path / "config.yaml"
        config.write_text(
            'paths:\n  data_dir: "/tmp/data"\ntypecodes:\n  - A320\n'
            + """\
selected_params:
  mach:
    {sigma_s: 8.0, sigma_r: 0.01, n_passes: 2, slope_tol: 3.0e-4,
     flat_tol: 5.0e-2, min_len: 15}
  cas:
    {cutoff_s: 180.0, sigma_s: 8.0, sigma_r: 3.0, n_passes: 2, slope_tol: 0.09,
     flat_tol: 20.0, min_len: 5}
  vz: {sigma_s: 6.0, sigma_r: 100.0, slope_tol: 50.0, flat_tol: 100.0, min_len: 10}
  alt: {sigma_s: 6.0, sigma_r: 20.0, n_passes: 2, tol_ftmin: 150.0, min_len: 6}
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002
    slope_tol: 1.2e-3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
    tol: 0.01
"""
        )
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params.gamma.tol == 0.01
        assert cfg.selected_params.gamma.min_len == 10
        assert cfg.selected_params.gamma.use_alt is False
        assert cfg.selected_params.gamma.smooth_window == 5
        assert cfg.selected_params.gamma.smooth_method == "savgol"


class TestSelectedParamConfigYaml:
    """YAML-loading tests for SelectedParamConfig overrides."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        """YAML with custom mach.tol=0.01 overrides the legacy default."""
        config = tmp_path / "config.yaml"
        config.write_text(
            'paths:\n  data_dir: "/tmp/data"\ntypecodes:\n  - A320\n'
            + """\
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01
    n_passes: 2
    slope_tol: 3.0e-4
    flat_tol: 5.0e-2
    min_len: 15
    tol: 0.01
  cas:
    {cutoff_s: 180.0, sigma_s: 8.0, sigma_r: 3.0, n_passes: 2, slope_tol: 0.09,
     flat_tol: 20.0, min_len: 5}
  vz: {sigma_s: 6.0, sigma_r: 100.0, slope_tol: 50.0, flat_tol: 100.0, min_len: 10}
  alt: {sigma_s: 6.0, sigma_r: 20.0, n_passes: 2, tol_ftmin: 150.0, min_len: 6}
  gamma:
    {sigma_s: 6.0, sigma_r: 0.002, slope_tol: 1.2e-3, flat_tol: 2.0e-3,
     abs_min: 5.0e-3, min_len: 10}
"""
        )
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params.mach.tol == 0.01
        # The declared calibration is what the model carries.
        assert cfg.selected_params.mach.min_len == 15

    def test_legacy_savgol_fields_keep_defaults(
        self, tmp_path: Path, make_config: Callable[..., Path]
    ) -> None:
        """Legacy savgol fields keep their defaults across every channel.

        They are tolerances of a superseded detector, not a calibration
        result, so they are the one part of these models that may still
        default. Partial override itself stays covered by
        ``test_yaml_custom_mach_tol`` and ``test_yaml_partial_gamma_override``.
        """
        config = make_config(tmp_path / "config.yaml", "/tmp/data")
        cfg = PipelineConfig.from_yaml(config)
        assert cfg.selected_params.mach.tol == 0.0005
        assert cfg.selected_params.cas.tol == 0.75
        assert cfg.selected_params.vz.min_abs_value == 75
        assert cfg.selected_params.alt.min_abs_value == 25
        assert cfg.selected_params.gamma.tol == 0.002

    def test_omitted_selected_params_raises(self, tmp_path: Path) -> None:
        """A config with no selected_params section at all fails to load.

        Replaces a test that asserted the omitted section fell back to
        defaults — the behaviour the required fields exist to abolish.
        """
        config = tmp_path / "config.yaml"
        config.write_text('paths:\n  data_dir: "/tmp/data"\ntypecodes:\n  - A320\n')
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig.from_yaml(config)
        assert "selected_params" in str(exc_info.value)

    def test_empty_selected_params_names_every_channel(self, tmp_path: Path) -> None:
        """An empty selected_params section names each channel left undeclared.

        Replaces the second half of a test that asserted the empty section
        fell back to defaults.
        """
        config = tmp_path / "config.yaml"
        config.write_text(
            'paths:\n  data_dir: "/tmp/data"\ntypecodes:\n  - A320\nselected_params: {}\n'
        )
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig.from_yaml(config)
        # The error names each channel, so the fix is readable from the message.
        message = str(exc_info.value)
        for channel in ("mach", "cas", "vz", "alt", "gamma"):
            assert f"selected_params.{channel}" in message
        # tas has no bilateral detector, so it keeps its default and is not named.
        assert "selected_params.tas" not in message

    def test_declared_values_round_trip(self, tmp_path: Path, selected_params_yaml: str) -> None:
        """The calibration written in YAML is exactly what the model carries."""
        config = tmp_path / "config.yaml"
        config.write_text(
            'paths:\n  data_dir: "/tmp/data"\ntypecodes:\n  - A320\n' + selected_params_yaml
        )
        cfg = PipelineConfig.from_yaml(config)
        # opensky26 tbl.3 — the vz sigma_r that superseded the old 350 default.
        assert cfg.selected_params.vz.sigma_r == 100.0
        assert cfg.selected_params.vz.slope_tol == 50.0
        assert cfg.selected_params.mach.sigma_r == 0.01
        assert cfg.selected_params.gamma.abs_min == 5.0e-3

    def test_gamma_legacy_defaults_survive_a_declared_calibration(
        self,
        tmp_path: Path,
        make_config: Callable[..., Path],
    ) -> None:
        """Declaring the bilateral knobs leaves the legacy savgol fields at defaults."""
        config = make_config(tmp_path / "config.yaml", "/tmp/data")
        cfg = PipelineConfig.from_yaml(config)
        defaults = selected_param_config().gamma
        assert cfg.selected_params.gamma.smooth_method == defaults.smooth_method
        assert cfg.selected_params.gamma.smooth_window == defaults.smooth_window
