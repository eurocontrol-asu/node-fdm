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
        assert info.segment_filter_fn is not None
        assert info.architecture_import == "node_fdm.architectures.opensky"
        assert callable(info.preprocessing_fn)

    def test_resolve_qar(self) -> None:
        """Resolving 'qar' returns correct ArchitectureInfo."""
        info = resolve_architecture("qar")
        assert isinstance(info, ArchitectureInfo)
        assert info.name == "qar"
        assert len(info.x_cols) > 0
        assert info.segment_filter_fn is None  # QAR has no segment filter
        assert info.architecture_import == "node_fdm.architectures.qar"

    def test_resolve_unknown_raises(self) -> None:
        """Unknown architecture name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture: 'unknown'"):
            resolve_architecture("unknown")

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

    def test_qar_preprocessing_fn_callable(self) -> None:
        """QAR preprocessing function is callable."""
        info = resolve_architecture("qar")
        assert callable(info.preprocessing_fn)
