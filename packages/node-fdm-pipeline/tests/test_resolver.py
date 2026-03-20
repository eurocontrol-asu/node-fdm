"""Tests for ArchitectureResolver."""

from __future__ import annotations

import pytest

from node_fdm_pipeline.resolver import ArchitectureInfo, resolve_architecture


class TestResolveArchitecture:
    """Tests for resolve_architecture function."""

    def test_resolve_opensky(self) -> None:
        """Resolving 'opensky' returns correct ArchitectureInfo."""
        info = resolve_architecture("opensky")
        assert isinstance(info, ArchitectureInfo)
        assert info.name == "opensky_2025"
        assert len(info.x_cols) > 0
        assert len(info.u_cols) > 0
        assert len(info.e0_cols) > 0
        assert len(info.dx_cols) > 0
        assert info.segment_filter_fn is None
        assert info.preprocessing_fn is None
        assert info.architecture_import == "node_fdm.architectures.opensky"

    def test_resolve_qar(self) -> None:
        """Resolving 'qar' returns correct ArchitectureInfo."""
        info = resolve_architecture("qar")
        assert isinstance(info, ArchitectureInfo)
        assert info.name == "qar"
        assert len(info.x_cols) > 0
        assert info.segment_filter_fn is None
        assert info.preprocessing_fn is None
        assert info.architecture_import == "node_fdm.architectures.qar"

    def test_resolve_unknown_raises(self) -> None:
        """Unknown architecture name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            resolve_architecture("unknown")

    def test_resolve_opensky_v2(self) -> None:
        """Resolving 'opensky_v2' returns ArchitectureInfo with lateral columns."""
        info = resolve_architecture("opensky_v2")
        assert isinstance(info, ArchitectureInfo)
        assert info.name == "opensky_v2"
        assert "latitude" in info.x_cols
        assert "longitude" in info.x_cols
        assert "track_sel" in info.x_cols
        assert len(info.x_cols) > len(resolve_architecture("opensky").x_cols)
        assert info.architecture_import == "node_fdm.architectures.opensky_v2"
        assert info.preprocessing_fn is None

    def test_opensky_v1_unchanged(self) -> None:
        """opensky v1 resolver still returns same schema (backward compat)."""
        info = resolve_architecture("opensky")
        assert info.name == "opensky_2025"
        assert "latitude" not in info.x_cols
        assert "longitude" not in info.x_cols
        assert len(info.x_cols) == 4  # Original 4 state vars

    def test_architecture_info_frozen(self) -> None:
        """ArchitectureInfo is immutable (frozen dataclass)."""
        info = resolve_architecture("opensky")
        with pytest.raises(AttributeError):
            info.name = "modified"  # type: ignore[misc]

    def test_opensky_has_dx_cols(self) -> None:
        """OpenSky DX_COLS are tuples of (int, str)."""
        info = resolve_architecture("opensky")
        for sign, col in info.dx_cols:
            assert isinstance(sign, int)
            assert isinstance(col, str)

    def test_qar_preprocessing_fn_none(self) -> None:
        """QAR preprocessing function is None (v3 pipeline handles preprocessing)."""
        info = resolve_architecture("qar")
        assert info.preprocessing_fn is None

    def test_resolve_adsb(self) -> None:
        """Resolving 'adsb' returns correct ArchitectureInfo."""
        info = resolve_architecture("adsb")
        assert isinstance(info, ArchitectureInfo)
        assert info.name == "node_adsb_v1"
        assert len(info.x_cols) == 3
        assert len(info.u_cols) == 4
        assert len(info.e0_cols) == 2
        assert len(info.dx_cols) == 3
        assert info.segment_filter_fn is None
        assert info.preprocessing_fn is None

    def test_resolve_adsb_cols(self) -> None:
        """Resolving 'adsb' returns correct column lists from schemas.adsb."""
        info = resolve_architecture("adsb")
        assert info.x_cols == ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]
        assert info.u_cols[0] == "fdm_alt_target_m"
        assert info.e0_cols == ["fdm_long_wind_ms", "era_temp_K"]

    def test_resolve_adsb_import(self) -> None:
        """Resolving 'adsb' sets correct architecture_import path."""
        info = resolve_architecture("adsb")
        assert info.architecture_import == "node_fdm.architectures.adsb"

    def test_resolve_unknown_mentions_adsb(self) -> None:
        """Unknown architecture error message includes 'adsb' in supported list."""
        with pytest.raises(ValueError, match="adsb"):
            resolve_architecture("unknown")

    def test_resolve_opensky_and_adsb_coexist(self) -> None:
        """Both opensky and adsb architectures can be resolved without conflict."""
        opensky = resolve_architecture("opensky")
        adsb = resolve_architecture("adsb")
        assert opensky.name != adsb.name
        assert opensky.x_cols != adsb.x_cols
