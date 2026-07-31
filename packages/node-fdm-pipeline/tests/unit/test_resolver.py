"""Tests for ArchitectureResolver."""

from __future__ import annotations

import pytest

from node_fdm_pipeline.resolver import ArchitectureInfo, resolve_architecture


class TestResolveArchitecture:
    """Tests for resolve_architecture function."""

    def test_resolve_qar(self) -> None:
        """Resolving 'qar' returns correct ArchitectureInfo."""
        info = resolve_architecture("qar")
        assert isinstance(info, ArchitectureInfo)
        assert info.name == "qar"
        assert len(info.x_cols) > 0
        assert info.segment_filter_fn is None
        assert info.preprocessing_fn.__module__ == "node_fdm_models.preprocessing.qar"

    @pytest.mark.parametrize(
        "match",
        [
            pytest.param("Unknown architecture", id="error_prefix"),
            pytest.param("adsb", id="lists_supported"),
        ],
    )
    def test_resolve_unknown_raises(self, match: str) -> None:
        """Unknown architecture raises ValueError mentioning prefix and supported names."""
        with pytest.raises(ValueError, match=match):
            resolve_architecture("unknown")

    def test_architecture_info_frozen(self) -> None:
        """ArchitectureInfo is immutable (frozen dataclass)."""
        info = resolve_architecture("adsb")
        with pytest.raises(AttributeError):
            info.name = "modified"  # type: ignore[misc]

    def test_qar_preprocessing_fn_is_provider_owned(self) -> None:
        """QAR preprocessing resolves from its installed model provider."""
        info = resolve_architecture("qar")
        assert callable(info.preprocessing_fn)
        assert info.preprocessing_fn.__module__ == "node_fdm_models.preprocessing.qar"

    def test_resolve_adsb(self) -> None:
        """Resolving 'adsb' returns correct ArchitectureInfo."""
        info = resolve_architecture("adsb")
        assert isinstance(info, ArchitectureInfo)
        assert info.name == "node_adsb_v1"
        assert "raw_alt_m" in info.x_cols
        assert "fdm_gamma_rad" in info.x_cols
        assert "era_tas_ms" in info.x_cols
        assert "fdm_alt_target_m" in info.u_cols
        assert "fdm_tas_target_ms" in info.u_cols
        assert "fdm_gamma_target_rad" in info.u_cols
        assert "era_temp_K" in info.e0_cols
        assert "fdm_long_wind_ms" in info.e0_cols
        assert len(info.dx_cols) > 0
        assert info.segment_filter_fn is None
        assert callable(info.preprocessing_fn)
        assert info.preprocessing_fn.__module__ == "node_fdm_models.preprocessing.opensky"
