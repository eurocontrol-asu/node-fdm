"""Tests for node_fdm_data.meteo — haversine, Mach/CAS, TAS, ERA5 enrichment."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from node_fdm_data.meteo import compute_mach_and_cas, compute_tas, enrich_era5, haversine


class TestHaversine:
    """Great-circle distance."""

    def test_cdg_to_jfk(self) -> None:
        """CDG (49.0097°N, 2.5479°E) → JFK (40.6413°N, -73.7781°W)."""
        d = haversine(
            np.array([49.0097]),
            np.array([2.5479]),
            np.array([40.6413]),
            np.array([-73.7781]),
        )
        # Expected ≈ 5834 km
        assert d[0] / 1000 == pytest.approx(5834, abs=10)

    def test_same_point(self) -> None:
        d = haversine(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        assert d[0] == pytest.approx(0.0)

    def test_vectorised(self) -> None:
        lat1 = np.array([0.0, 49.0097])
        lon1 = np.array([0.0, 2.5479])
        lat2 = np.array([0.0, 40.6413])
        lon2 = np.array([0.0, -73.7781])
        d = haversine(lat1, lon1, lat2, lon2)
        assert d[0] == pytest.approx(0.0)
        assert d[1] / 1000 == pytest.approx(5834, abs=10)


class TestComputeMachAndCas:
    """Mach number and CAS derivation."""

    def test_compute_mach_from_tas(self) -> None:
        """TAS=460 kt, alt=35000 ft → Mach ≈ 0.78 (typical cruise)."""
        from node_fdm_data.physics.isa import isa_temperature

        alt_ft = 35_000.0
        alt_m = alt_ft * 0.3048
        temp = float(isa_temperature(alt_m))
        mach, cas_kt = compute_mach_and_cas(
            tas_kt=np.array([460.0]),
            alt_ft=np.array([alt_ft]),
            temp_k=np.array([temp]),
        )
        assert mach[0] == pytest.approx(0.78, abs=0.02)
        assert cas_kt[0] > 0

    def test_sea_level(self) -> None:
        """At sea level, TAS ≈ CAS (no compressibility correction)."""
        _mach, cas_kt = compute_mach_and_cas(
            tas_kt=np.array([100.0]),
            alt_ft=np.array([0.0]),
            temp_k=np.array([288.15]),
        )
        assert cas_kt[0] == pytest.approx(100.0, abs=1.0)

    def test_vectorised(self) -> None:
        _mach, cas = compute_mach_and_cas(
            tas_kt=np.array([100.0, 250.0]),
            alt_ft=np.array([0.0, 30_000.0]),
            temp_k=np.array([288.15, 228.71]),
        )
        assert len(_mach) == 2
        assert len(cas) == 2


class TestComputeTas:
    """TAS from groundspeed + wind (Polars expression)."""

    def test_no_wind(self) -> None:
        """With zero wind, TAS ≈ groundspeed."""
        df = pl.DataFrame(
            {
                "raw_gs_kt": [250.0],
                "raw_track_deg": [90.0],
                "era_u_wind_ms": [0.0],
                "era_v_wind_ms": [0.0],
            }
        )
        result = df.select(compute_tas().alias("tas"))
        assert result["tas"][0] == pytest.approx(250.0, abs=0.1)

    def test_headwind(self) -> None:
        """Headwind reduces TAS vs groundspeed (but here we reconstruct)."""
        df = pl.DataFrame(
            {
                "raw_gs_kt": [200.0],
                "raw_track_deg": [0.0],  # heading north
                "era_u_wind_ms": [0.0],
                "era_v_wind_ms": [-10.0],  # southerly wind (headwind) in m/s
            }
        )
        result = df.select(compute_tas().alias("tas"))
        # TAS should differ from groundspeed
        assert result["tas"][0] > 0


def _make_base_df(n: int = 3) -> pl.DataFrame:
    """Build a minimal DataFrame with raw_* columns for enrich_era5 tests."""
    return pl.DataFrame(
        {
            "raw_lat_deg": [48.8566] * n,
            "raw_lon_deg": [2.3522] * n,
            "raw_alt_ft": [35_000.0] * n,
            "raw_timestamp": [datetime(2024, 6, 15, 12, 0, 0)] * n,
            "raw_gs_kt": [450.0] * n,
            "raw_track_deg": [90.0] * n,
            "bds_tas_kt": [460.0] * n,
            "bds_mach": [0.78] * n,
        }
    )


def _mock_arco_grid(
    extra_cols: dict[str, list[float]] | None = None,
) -> MagicMock:
    """Return a mock ArcoEra5 whose interpolate adds ERA5 weather columns."""

    def _interpolate(pdf: pd.DataFrame) -> pd.DataFrame:
        n = len(pdf)
        pdf["temperature"] = (
            extra_cols["temperature"]
            if extra_cols and "temperature" in extra_cols
            else [220.0] * n
        )
        pdf["u_component_of_wind"] = (
            extra_cols["u_component_of_wind"]
            if extra_cols and "u_component_of_wind" in extra_cols
            else [5.0] * n
        )
        pdf["v_component_of_wind"] = (
            extra_cols["v_component_of_wind"]
            if extra_cols and "v_component_of_wind" in extra_cols
            else [-3.0] * n
        )
        return pdf

    mock = MagicMock()
    mock.interpolate = _interpolate
    return mock


class TestEnrichEra5Columns:
    """ERA5 enrichment adds the expected weather columns."""

    def test_enrich_era5_columns(self) -> None:
        df = _make_base_df()
        result = enrich_era5(df, _mock_arco_grid())

        assert "era_temp_K" in result.columns
        assert "era_u_wind_ms" in result.columns
        assert "era_v_wind_ms" in result.columns
        assert result["era_temp_K"][0] == pytest.approx(220.0)
        assert result["era_u_wind_ms"][0] == pytest.approx(5.0)
        assert result["era_v_wind_ms"][0] == pytest.approx(-3.0)

    def test_enrich_era_tas(self) -> None:
        """era_tas_kt is computed from GS + ERA5 wind."""
        df = _make_base_df(1)
        result = enrich_era5(df, _mock_arco_grid())

        assert "era_tas_kt" in result.columns
        # With wind present, TAS ≠ GS
        assert result["era_tas_kt"][0] > 0
        assert result["era_tas_kt"][0] != pytest.approx(450.0, abs=0.01)

    def test_enrich_era_mach_cas(self) -> None:
        """era_mach and era_cas_kt are computed consistently."""
        df = _make_base_df(1)
        result = enrich_era5(df, _mock_arco_grid())

        assert "era_mach" in result.columns
        assert "era_cas_kt" in result.columns
        # Typical cruise: Mach should be in plausible range
        assert 0.5 < result["era_mach"][0] < 1.1
        assert result["era_cas_kt"][0] > 0

    def test_enrich_preserves_bds(self) -> None:
        """BDS columns must not be overwritten."""
        df = _make_base_df(1)
        result = enrich_era5(df, _mock_arco_grid())

        assert result["bds_tas_kt"][0] == pytest.approx(460.0)
        assert result["bds_mach"][0] == pytest.approx(0.78)


class TestEnrichEra5EdgeCases:
    """Edge cases for ERA5 enrichment."""

    def test_temperature_nan(self) -> None:
        """NaN temperature → era_mach and era_cas_kt are NaN (no crash)."""
        df = _make_base_df(1)
        mock = _mock_arco_grid(
            extra_cols={
                "temperature": [float("nan")],
                "u_component_of_wind": [5.0],
                "v_component_of_wind": [-3.0],
            }
        )
        result = enrich_era5(df, mock)

        assert np.isnan(result["era_mach"][0])
        assert np.isnan(result["era_cas_kt"][0])

    def test_gs_zero(self) -> None:
        """GS=0 (aircraft on ground) → era_tas_kt ≈ wind speed, no crash."""
        df = pl.DataFrame(
            {
                "raw_lat_deg": [48.8566],
                "raw_lon_deg": [2.3522],
                "raw_alt_ft": [0.0],
                "raw_timestamp": [datetime(2024, 6, 15, 12, 0, 0)],
                "raw_gs_kt": [0.0],
                "raw_track_deg": [0.0],
            }
        )
        mock = _mock_arco_grid(
            extra_cols={
                "temperature": [288.15],
                "u_component_of_wind": [5.0],
                "v_component_of_wind": [0.0],
            }
        )
        result = enrich_era5(df, mock)

        # TAS ≈ wind magnitude in kt (5 m/s ≈ 9.7 kt)
        assert result["era_tas_kt"][0] == pytest.approx(5.0 * 1.94384, abs=0.5)
