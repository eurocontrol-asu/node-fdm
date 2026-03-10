"""Tests for opensky_v2 schema column lists."""

from __future__ import annotations

from node_fdm_data.schemas import opensky, opensky_v2


class TestOpenskyV2Schema:
    """Tests for the opensky_v2 schema lateral extensions."""

    def test_x_cols_includes_lateral(self) -> None:
        """X_COLS includes latitude, longitude, track_sel."""
        assert "latitude" in opensky_v2.X_COLS
        assert "longitude" in opensky_v2.X_COLS
        assert "track_sel" in opensky_v2.X_COLS

    def test_x_cols_all_strings(self) -> None:
        """All X_COLS entries are strings."""
        for col in opensky_v2.X_COLS:
            assert isinstance(col, str)

    def test_e1_cols_includes_track(self) -> None:
        """E1_COLS includes track."""
        assert "track" in opensky_v2.E1_COLS

    def test_dx_cols_includes_d_track(self) -> None:
        """DX_COLS includes d_track derivative."""
        dx_names = [name for _, name in opensky_v2.DX_COLS]
        assert "d_track" in dx_names

    def test_v2_extends_v1(self) -> None:
        """V2 X_COLS is a superset of V1 X_COLS."""
        for col in opensky.X_COLS:
            assert col in opensky_v2.X_COLS

    def test_u_cols_unchanged(self) -> None:
        """U_COLS (control inputs) same between V1 and V2."""
        assert opensky_v2.U_COLS == opensky.U_COLS

    def test_e0_cols_unchanged(self) -> None:
        """E0_COLS (environment at t0) same between V1 and V2."""
        assert opensky_v2.E0_COLS == opensky.E0_COLS
