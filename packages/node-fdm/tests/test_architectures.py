"""Tests for opensky_v2 architecture registration."""

from __future__ import annotations

import pytest

from node_fdm.architectures.registry import get


class TestOpenskyV2Architecture:
    """Tests for opensky_v2 architecture auto-registration."""

    def test_resolve_opensky_v2(self) -> None:
        """'opensky_v2' is registered and resolvable."""
        # Force import to trigger auto-registration
        import node_fdm.architectures.opensky_v2  # noqa: F401

        spec = get("opensky_v2")
        assert spec.name == "opensky_v2"
        assert "latitude" in spec.x_cols
        assert "longitude" in spec.x_cols
        assert "track_sel" in spec.x_cols
        assert len(spec.layers) == 2

    def test_opensky_v1_unchanged(self) -> None:
        """'opensky_2025' architecture is still registered and untouched."""
        import node_fdm.architectures.opensky  # noqa: F401

        spec = get("opensky_2025")
        assert spec.name == "opensky_2025"
        assert "latitude" not in spec.x_cols
        assert len(spec.x_cols) == 4

    def test_unknown_architecture_raises(self) -> None:
        """Unknown architecture name gives clear error."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            get("opensky_v99")

    def test_v2_has_lateral_derivatives(self) -> None:
        """V2 DX_COLS includes d_track."""
        import node_fdm.architectures.opensky_v2  # noqa: F401

        spec = get("opensky_v2")
        dx_names = [name for _, name in spec.dx_cols]
        assert "d_track" in dx_names
