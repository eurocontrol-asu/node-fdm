"""Tests for ADS-B v1 schema column lists."""

from __future__ import annotations

from node_fdm_data.schemas import adsb, opensky


class TestAdsbSchema:
    """Tests for the adsb schema simplifications vs opensky_2025."""

    def test_x_cols_no_distance(self) -> None:
        """X_COLS removes fdm_distance_cum_m, keeps 3 state variables."""
        assert "fdm_distance_cum_m" not in adsb.X_COLS
        assert len(adsb.X_COLS) == 3

    def test_x_cols_content(self) -> None:
        """X_COLS contains the expected state variables."""
        assert adsb.X_COLS == ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]

    def test_u_cols_alt_target(self) -> None:
        """U_COLS[0] is fdm_alt_target_m (robust, never NaN)."""
        assert adsb.U_COLS[0] == "fdm_alt_target_m"

    def test_e0_cols_no_airport_dist(self) -> None:
        """E0_COLS has 2 variables, no airport distances."""
        assert len(adsb.E0_COLS) == 2
        assert "fdm_adep_dist_m" not in adsb.E0_COLS
        assert "fdm_ades_dist_m" not in adsb.E0_COLS

    def test_e1_cols_identical_to_opensky(self) -> None:
        """E1_COLS is identical to opensky_2025."""
        assert adsb.E1_COLS == opensky.E1_COLS

    def test_dx_cols_no_gs(self) -> None:
        """DX_COLS does not include raw_gs_ms."""
        dx_names = [name for _, name in adsb.DX_COLS]
        assert "raw_gs_ms" not in dx_names

    def test_dx_cols_content(self) -> None:
        """DX_COLS has 3 derivatives with correct signs."""
        assert adsb.DX_COLS == [
            (1, "fdm_d_vz_ms"),
            (1, "fdm_d_gamma_rads"),
            (1, "fdm_d_tas_ms"),
        ]


class TestBothSchemasCoexist:
    """Both schemas can be loaded without conflict."""

    def test_both_loaded(self) -> None:
        """Importing both schemas does not raise."""
        assert len(opensky.X_COLS) == 4
        assert len(adsb.X_COLS) == 3
