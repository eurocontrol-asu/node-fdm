"""Unit tests for PipelineConfig models (pure, no I/O)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from node_fdm_pipeline import config as config_module
from node_fdm_pipeline.config import (
    GammaFilterConfig,
    LateralDetectionConfig,
    PathsConfig,
    PipelineConfig,
    SelectedParamConfig,
)


class TestFleetRunConfig:
    """Unit tests for the deployment-owned fleet run settings."""

    def test_zero_or_missing_lease_ttl_is_rejected(self) -> None:
        """AC2: lease_ttl_s is required and must be strictly positive."""
        with pytest.raises(ValidationError, match="lease_ttl_s"):
            config_module.FleetRunConfig(
                lease_path=Path("/tmp/trino.lease"),
                lease_ttl_s=0,
                disk_min_gib=1.0,
            )

        with pytest.raises(ValidationError, match="lease_ttl_s"):
            config_module.FleetRunConfig.model_validate(
                {
                    "lease_path": Path("/tmp/trino.lease"),
                    "disk_min_gib": 1.0,
                }
            )

    def test_lease_path_is_expanded_and_absolute(self) -> None:
        """AC3: a home-relative lease_path is exposed as an expanded absolute Path."""
        cfg = config_module.FleetRunConfig(
            lease_path=Path("~/shared/trino.lease"),
            lease_ttl_s=60,
            disk_min_gib=1.0,
        )

        assert cfg.lease_path == Path("~/shared/trino.lease").expanduser().resolve()
        assert cfg.lease_path.is_absolute() is True
        assert "~" not in str(cfg.lease_path)


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

    def test_calibrated_fields_are_required(self) -> None:
        """Constructing without the calibrated hyper-parameters fails, naming each.

        Replaces ``test_default_fields``, which asserted on defaults these
        fields no longer carry.
        """
        with pytest.raises(ValidationError) as exc_info:
            GammaFilterConfig()  # type: ignore[call-arg]
        message = str(exc_info.value)
        for field in ("sigma_s", "sigma_r", "slope_tol", "flat_tol", "abs_min", "min_len"):
            assert field in message

    def test_declared_fields_round_trip_and_legacy_fields_default(self) -> None:
        """Supplied values round-trip; ``mode`` and savgol fields keep defaults."""
        cfg = GammaFilterConfig(
            sigma_s=6.0,
            sigma_r=0.002,
            slope_tol=1.2e-3,
            flat_tol=2.0e-3,
            abs_min=5.0e-3,
            min_len=10,
        )
        assert cfg.sigma_r == 0.002
        assert cfg.slope_tol == 1.2e-3
        assert cfg.abs_min == 5.0e-3
        assert cfg.min_len == 10
        # Not part of the calibration — still defaulted.
        assert cfg.mode == "bilateral_gamma"
        assert cfg.tol == 0.002
        assert cfg.use_alt is False
        assert cfg.smooth_window == 5
        assert cfg.smooth_method == "savgol"

    def test_frozen(self) -> None:
        """GammaFilterConfig is immutable (frozen=True)."""
        cfg = GammaFilterConfig(
            sigma_s=6.0,
            sigma_r=0.002,
            slope_tol=1.2e-3,
            flat_tol=2.0e-3,
            abs_min=5.0e-3,
            min_len=10,
        )
        with pytest.raises(ValidationError):
            cfg.tol = 0.5  # type: ignore[misc]

    def test_custom_values(self) -> None:
        """GammaFilterConfig accepts custom values for its legacy fields."""
        cfg = GammaFilterConfig(
            sigma_s=6.0,
            sigma_r=0.002,
            slope_tol=1.2e-3,
            flat_tol=2.0e-3,
            abs_min=5.0e-3,
            min_len=30,
            tol=0.01,
            use_alt=True,
        )
        assert cfg.tol == 0.01
        assert cfg.min_len == 30
        assert cfg.use_alt is True


class TestSelectedParamConfigUnit:
    """Unit tests for SelectedParamConfig — the required/defaulted split."""

    def test_detector_channels_are_required(self) -> None:
        """Every bilateral channel must be declared; the error names each one.

        Replaces ``test_default_values``, which asserted on the defaults this
        change removed. ``tas`` is deliberately absent from the expected
        errors — it has no bilateral detector and keeps its default.
        """
        with pytest.raises(ValidationError) as exc_info:
            SelectedParamConfig()  # type: ignore[call-arg]
        message = str(exc_info.value)
        for channel in ("mach", "cas", "vz", "alt", "gamma"):
            assert channel in message
        assert "\ntas\n" not in message

    def test_declared_channels_round_trip(
        self, selected_params: dict[str, dict[str, float | int]]
    ) -> None:
        """Declared calibration round-trips; legacy savgol fields keep defaults."""
        cfg = SelectedParamConfig(**selected_params)  # type: ignore[arg-type]
        # The calibrated values are exactly what was declared.
        assert cfg.mach.sigma_r == 0.01
        assert cfg.cas.slope_tol == 0.09
        assert cfg.vz.sigma_r == 100.0
        assert cfg.alt.tol_ftmin == 150.0
        assert cfg.gamma.abs_min == 5.0e-3
        # Legacy savgol fields still default.
        assert cfg.mach.tol == 0.0005
        assert cfg.mach.alt_threshold == 15000
        assert cfg.mach.use_alt is True
        assert cfg.cas.tol == 0.75
        assert cfg.cas.smooth_method == "savgol"
        assert cfg.vz.min_abs_value == 75
        assert cfg.alt.tol == 25
        assert cfg.gamma.tol == 0.002
        assert cfg.gamma.smooth_window == 5

    def test_tas_keeps_its_default(
        self, selected_params: dict[str, dict[str, float | int]]
    ) -> None:
        """``tas`` has no bilateral detector, so it stays optional."""
        cfg = SelectedParamConfig(**selected_params)  # type: ignore[arg-type]
        assert cfg.tas.tol == 0.75
        assert cfg.tas.min_len == 20
        assert cfg.mach_min_value == 0.5


class TestLateralDetectionConfigUnit:
    """Unit tests for LateralDetectionConfig — V3 turn-detector hyperparams."""

    def test_lateral_detection_defaults(self) -> None:
        """AC1: default field values match the V3 baseline numerics."""
        cfg = LateralDetectionConfig()
        assert cfg.bilateral_sigma_s == 8.0
        assert cfg.bilateral_sigma_r == 0.01
        assert cfg.bilateral_passes == 2
        assert cfg.rate_threshold == 0.05

    def test_lateral_detection_negative_rejected(self) -> None:
        """AC8: negative rate_threshold raises ValidationError mentioning the field."""
        with pytest.raises(ValidationError) as exc_info:
            LateralDetectionConfig(rate_threshold=-0.1)
        assert "rate_threshold" in str(exc_info.value)

    def test_lateral_detection_zero_passes_rejected(self) -> None:
        """AC8: zero bilateral_passes raises ValidationError (passes >= 1)."""
        with pytest.raises(ValidationError):
            LateralDetectionConfig(bilateral_passes=0)

    def test_lateral_detection_zero_sigma_rejected(self) -> None:
        """AC8: zero bilateral_sigma_s raises ValidationError (must be > 0)."""
        with pytest.raises(ValidationError):
            LateralDetectionConfig(bilateral_sigma_s=0.0)


class TestPipelineConfigLateralBlock:
    """Round-trip: PipelineConfig with / without lateral_detection block."""

    @staticmethod
    def _minimal_yaml(
        selected_params: dict[str, dict[str, float | int]],
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        data: dict[str, object] = {
            "paths": {"data_dir": "/tmp/data"},
            "typecodes": ["A320"],
            "selected_params": selected_params,
        }
        if extra:
            data.update(extra)
        return data

    def test_pipeline_config_block_optional(
        self, selected_params: dict[str, dict[str, float | int]]
    ) -> None:
        """AC7: omitting lateral_detection yields the documented defaults."""
        raw = yaml.safe_dump(self._minimal_yaml(selected_params))
        cfg = PipelineConfig.model_validate(yaml.safe_load(raw))
        assert cfg.lateral_detection.rate_threshold == 0.05
        assert cfg.lateral_detection.bilateral_sigma_s == 8.0
        assert cfg.lateral_detection.bilateral_sigma_r == 0.01
        assert cfg.lateral_detection.bilateral_passes == 2

    def test_pipeline_config_block_overrides(
        self, selected_params: dict[str, dict[str, float | int]]
    ) -> None:
        """AC7: overriding rate_threshold preserves other defaults."""
        raw = yaml.safe_dump(
            self._minimal_yaml(selected_params, {"lateral_detection": {"rate_threshold": 0.10}})
        )
        cfg = PipelineConfig.model_validate(yaml.safe_load(raw))
        assert cfg.lateral_detection.rate_threshold == 0.10
        assert cfg.lateral_detection.bilateral_sigma_s == 8.0
        assert cfg.lateral_detection.bilateral_passes == 2


def test_explicit_data_root_rebases_cohort_and_shared_weather_cache(
    tmp_path: Path, make_config: Callable[..., Path]
) -> None:
    config = make_config(tmp_path / "config.yaml", "/old/machine/A320neo__PW1100G")
    root = tmp_path / "portable"

    cfg = PipelineConfig.from_yaml(config, data_root=root)

    assert cfg.paths.data_dir == root.resolve() / "A320neo__PW1100G"
    assert cfg.paths.resolve("era5_cache_dir") == root.resolve() / "era5_cache"


def test_node_fdm_data_environment_rebases_config(
    tmp_path: Path,
    make_config: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path / "config.yaml", "/old/machine/E170")
    root = tmp_path / "environment-root"
    monkeypatch.setenv("NODE_FDM_DATA", str(root))

    cfg = PipelineConfig.from_yaml(config)

    assert cfg.paths.data_dir == root.resolve() / "E170"
