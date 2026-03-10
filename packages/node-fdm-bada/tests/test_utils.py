"""Tests for BADA utility functions — CAS↔Mach, TAS↔CAS conversions."""

from __future__ import annotations

import numpy as np
import pytest

from node_fdm_bada.utils import cas_to_mach, get_phase, mach_to_cas, ms_to_kt, tas_to_cas

# ---------------------------------------------------------------------------
# ms_to_kt
# ---------------------------------------------------------------------------


class TestMsToKt:
    """Tests for m/s → knots conversion."""

    def test_known_value(self) -> None:
        assert ms_to_kt(1.0) == pytest.approx(1.9438, rel=1e-3)

    def test_zero(self) -> None:
        assert ms_to_kt(0.0) == 0.0


# ---------------------------------------------------------------------------
# get_phase
# ---------------------------------------------------------------------------


class TestGetPhase:
    """Tests for flight-phase inference."""

    def test_cruise(self) -> None:
        assert get_phase(30_000.0, 30_030.0) == "Cruise"

    def test_climb(self) -> None:
        assert get_phase(20_000.0, 35_000.0) == "Climb"

    def test_descent(self) -> None:
        assert get_phase(35_000.0, 20_000.0) == "Descent"

    def test_threshold_boundary(self) -> None:
        # Exactly at threshold → should still be Cruise
        assert get_phase(30_000.0, 30_049.0) == "Cruise"
        # Just above threshold → Climb
        assert get_phase(30_000.0, 30_051.0) == "Climb"


# ---------------------------------------------------------------------------
# cas_to_mach
# ---------------------------------------------------------------------------


class TestCasToMach:
    """Tests for CAS → Mach conversion."""

    def test_cas_250kt_30000ft(self) -> None:
        """CAS=250 kt, alt=30000 ft → Mach ≈ 0.668 (ISA)."""
        cas_ms = 250.0 * 1852.0 / 3600.0  # 250 kt → m/s
        alt_m = 30_000.0 * 0.3048  # 30000 ft → m
        mach = float(cas_to_mach(cas_ms, alt_m))
        # Correct ISA value is ≈ 0.668 (ticket suggested 0.78, which is too high)
        assert mach == pytest.approx(0.668, abs=0.02)

    def test_sea_level(self) -> None:
        """Edge case: CAS→Mach at sea level should give valid result."""
        cas_ms = 250.0 * 1852.0 / 3600.0
        mach = float(cas_to_mach(cas_ms, 0.0))
        assert 0.0 < mach < 1.0

    def test_array_input(self) -> None:
        """Vectorized input should work."""
        cas_ms = np.array([128.6, 154.3])
        alt_m = np.array([5000.0, 8000.0])
        result = cas_to_mach(cas_ms, alt_m)
        assert result.shape == (2,)
        assert np.all(result > 0)


# ---------------------------------------------------------------------------
# mach_to_cas
# ---------------------------------------------------------------------------


class TestMachToCas:
    """Tests for Mach → CAS conversion."""

    def test_mach_080_35000ft(self) -> None:
        """AC5 test spec: Mach=0.80, alt=35000 ft → valid CAS."""
        alt_m = 35_000.0 * 0.3048
        cas = float(mach_to_cas(0.80, alt_m))
        # Should be a meaningful CAS in m/s (roughly 230-260 kt range)
        cas_kt = cas * 3600.0 / 1852.0
        assert 200.0 < cas_kt < 300.0

    def test_roundtrip_cas_mach_cas(self) -> None:
        """CAS → Mach → CAS should recover the original CAS."""
        cas_ms_orig = 250.0 * 1852.0 / 3600.0
        alt_m = 30_000.0 * 0.3048
        mach = cas_to_mach(cas_ms_orig, alt_m)
        cas_ms_recovered = float(mach_to_cas(mach, alt_m))
        assert cas_ms_recovered == pytest.approx(cas_ms_orig, rel=1e-6)


# ---------------------------------------------------------------------------
# tas_to_cas
# ---------------------------------------------------------------------------


class TestTasToCas:
    """Tests for TAS → CAS conversion."""

    def test_known_conditions(self) -> None:
        """TAS at cruise altitude should give a valid CAS."""
        tas_ms = 240.0  # ~466 kt
        alt_m = 10_000.0  # ~32800 ft
        temp_k = 223.0  # typical at that altitude
        cas = float(tas_to_cas(tas_ms, alt_m, temp_k))
        cas_kt = cas * 3600.0 / 1852.0
        assert 100.0 < cas_kt < 400.0

    def test_sea_level_isa(self) -> None:
        """At sea level with ISA temp, TAS ≈ CAS."""
        tas_ms = 100.0
        temp_k = 288.15  # ISA sea-level temperature
        cas = float(tas_to_cas(tas_ms, 0.0, temp_k))
        # At sea level, TAS ≈ CAS
        assert cas == pytest.approx(tas_ms, rel=0.01)
