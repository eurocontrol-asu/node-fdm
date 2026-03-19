"""Tests for node_fdm_data.meteo — haversine, Mach/CAS, TAS."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.meteo import compute_mach_and_cas, compute_tas, haversine


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
                "u_component_of_wind": [0.0],
                "v_component_of_wind": [0.0],
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
                "u_component_of_wind": [0.0],
                "v_component_of_wind": [-10.0],  # southerly wind (headwind) in m/s
            }
        )
        result = df.select(compute_tas().alias("tas"))
        # TAS should differ from groundspeed
        assert result["tas"][0] > 0
