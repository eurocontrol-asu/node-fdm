"""Tests for node_fdm_data.preprocessing.derive — pipeline v3 étape 4."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.physics.constants import FTMIN, KT
from node_fdm_data.preprocessing.derive import derive_columns


class TestDeriveGamma:
    """fdm_gamma_rad = arcsin(raw_vz_ftmin * FTMIN / (bds_tas_from_cas_kt * KT))."""

    def test_derive_gamma(self) -> None:
        """TAS=250kt, vz=1000ft/min → correct fdm_gamma_rad."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [1000.0],
                "bds_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_sel_alt_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
            }
        )
        result = derive_columns(df)
        assert "fdm_gamma_rad" in result.columns
        expected = np.arcsin((1000 * FTMIN) / (250 * KT))
        assert result["fdm_gamma_rad"][0] == pytest.approx(expected)

    def test_gamma_tas_near_zero(self) -> None:
        """TAS ≈ 0 → fdm_gamma_rad is clipped, no inf."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [1000.0],
                "bds_tas_from_cas_kt": [0.001],
                "raw_gs_kt": [0.0],
                "raw_alt_ft": [1000.0],
                "bds_mcp_sel_alt_ft": [2000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
            }
        )
        result = derive_columns(df)
        gamma = result["fdm_gamma_rad"][0]
        assert np.isfinite(gamma)
        # Clipped ratio → arcsin(1.0) = π/2
        assert abs(gamma) <= np.pi / 2 + 1e-6


class TestDeriveLongWind:
    """fdm_long_wind_kt = bds_tas_from_cas_kt - raw_gs_kt."""

    def test_derive_long_wind(self) -> None:
        """TAS=250kt, GS=240kt → fdm_long_wind_kt = 10.0."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "bds_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_sel_alt_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
            }
        )
        result = derive_columns(df)
        assert result["fdm_long_wind_kt"][0] == pytest.approx(10.0)


class TestDeriveAltDiff:
    """fdm_alt_diff_ft = bds_mcp_sel_alt_ft - raw_alt_ft."""

    def test_derive_alt_diff(self) -> None:
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "bds_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_sel_alt_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
            }
        )
        result = derive_columns(df)
        assert result["fdm_alt_diff_ft"][0] == pytest.approx(1000.0)


class TestDeriveDistanceCum:
    """fdm_distance_cum_m = haversine cumulative per flight."""

    def test_derive_distance_cum(self) -> None:
        """3 GPS points → correct cumulative haversine distance."""
        from node_fdm_data.meteo import haversine

        lats = [48.0, 48.01, 48.02]
        lons = [2.0, 2.0, 2.0]
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0] * 3,
                "bds_tas_from_cas_kt": [250.0] * 3,
                "raw_gs_kt": [240.0] * 3,
                "raw_alt_ft": [35000.0] * 3,
                "bds_mcp_sel_alt_ft": [36000.0] * 3,
                "raw_lat_deg": lats,
                "raw_lon_deg": lons,
                "meta_flight_id": ["F001"] * 3,
            }
        )
        result = derive_columns(df)
        assert "fdm_distance_cum_m" in result.columns
        d = result["fdm_distance_cum_m"].to_list()
        assert d[0] == 0.0

        # Verify against manual haversine
        d01 = haversine(
            np.array([lats[0]]),
            np.array([lons[0]]),
            np.array([lats[1]]),
            np.array([lons[1]]),
        )[0]
        d12 = haversine(
            np.array([lats[1]]),
            np.array([lons[1]]),
            np.array([lats[2]]),
            np.array([lons[2]]),
        )[0]
        assert d[1] == pytest.approx(d01, rel=1e-6)
        assert d[2] == pytest.approx(d01 + d12, rel=1e-6)

    def test_derive_per_flight(self) -> None:
        """2 flights in DataFrame → distance resets to 0 for each flight."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0] * 4,
                "bds_tas_from_cas_kt": [250.0] * 4,
                "raw_gs_kt": [240.0] * 4,
                "raw_alt_ft": [35000.0] * 4,
                "bds_mcp_sel_alt_ft": [36000.0] * 4,
                "raw_lat_deg": [48.0, 48.01, 49.0, 49.01],
                "raw_lon_deg": [2.0, 2.0, 3.0, 3.0],
                "meta_flight_id": ["F001", "F001", "F002", "F002"],
            }
        )
        result = derive_columns(df)
        d = result["fdm_distance_cum_m"].to_list()
        # Flight F001 starts at 0
        assert d[0] == 0.0
        assert d[1] > 0.0
        # Flight F002 restarts at 0
        assert d[2] == 0.0
        assert d[3] > 0.0


class TestDeriveAirportDistances:
    """fdm_adep_dist_nm / fdm_ades_dist_nm via airport_coords."""

    def test_airport_distances_with_coords(self) -> None:
        """Airport distances computed when airport_coords provided."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "bds_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_sel_alt_ft": [36000.0],
                "raw_lat_deg": [48.8566],
                "raw_lon_deg": [2.3522],
                "meta_flight_id": ["F001"],
                "meta_departure": ["LFPG"],
                "meta_arrival": ["KJFK"],
            }
        )
        # CDG ≈ (49.0097, 2.5479), JFK ≈ (40.6413, -73.7781)
        airport_coords = {
            "LFPG": (49.0097, 2.5479),
            "KJFK": (40.6413, -73.7781),
        }
        result = derive_columns(df, airport_coords=airport_coords)
        assert "fdm_adep_dist_nm" in result.columns
        assert "fdm_ades_dist_nm" in result.columns
        # ADEP (CDG) is close → small distance
        assert result["fdm_adep_dist_nm"][0] < 20.0
        # ADES (JFK) is far → large distance
        assert result["fdm_ades_dist_nm"][0] > 3000.0

    def test_airport_distances_null_departure(self) -> None:
        """meta_departure = null → fdm_adep_dist_nm = NaN."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "bds_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_sel_alt_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
                "meta_departure": [None],
                "meta_arrival": ["KJFK"],
            }
        )
        airport_coords = {"KJFK": (40.6413, -73.7781)}
        result = derive_columns(df, airport_coords=airport_coords)
        assert result["fdm_adep_dist_nm"][0] is None or np.isnan(result["fdm_adep_dist_nm"][0])

    def test_airport_distances_no_coords(self) -> None:
        """No airport_coords → columns filled with null."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "bds_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_sel_alt_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
                "meta_departure": ["LFPG"],
                "meta_arrival": ["KJFK"],
            }
        )
        result = derive_columns(df)
        assert "fdm_adep_dist_nm" in result.columns
        assert "fdm_ades_dist_nm" in result.columns
        assert result["fdm_adep_dist_nm"].null_count() == 1
        assert result["fdm_ades_dist_nm"].null_count() == 1
