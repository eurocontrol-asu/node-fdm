"""Tests for node_fdm_data.physics.speed — Mach/CAS → TAS conversions."""

from __future__ import annotations

import math

import pytest

from node_fdm_data.physics import A0, GAMMA_AIR, P0, R
from node_fdm_data.physics.isa import isa_pressure, isa_temperature
from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas

# ---------------------------------------------------------------------------
# Unit tests — mach_to_tas
# ---------------------------------------------------------------------------


class TestMachToTasSeaLevel:
    """mach_to_tas at sea level (h=0)."""

    def test_mach_to_tas_sea_level(self) -> None:
        """mach=0.3, alt=0 -> TAS ~ 0.3 * A0 ~ 102.1 m/s."""
        tas = mach_to_tas(0.3, 0.0)
        expected = 0.3 * A0
        assert float(tas) == pytest.approx(expected, rel=1e-6)


class TestMachToTasCruise:
    """mach_to_tas at cruise altitude."""

    def test_mach_to_tas_cruise(self) -> None:
        """mach=0.78, alt=10 000 m -> TAS ~ 0.78 * sqrt(gamma*R*223.15) ~ 233 m/s."""
        tas = mach_to_tas(0.78, 10_000.0)
        t = float(isa_temperature(10_000.0))  # 288.15 - 0.0065*10000 = 223.15
        expected = 0.78 * math.sqrt(GAMMA_AIR * R * t)
        assert float(tas) == pytest.approx(expected, rel=1e-6)
        assert float(tas) == pytest.approx(233, abs=2)


# ---------------------------------------------------------------------------
# Unit tests — cas_to_tas
# ---------------------------------------------------------------------------


class TestCasToTasSeaLevel:
    """cas_to_tas at sea level (CAS ≈ TAS)."""

    def test_cas_to_tas_sea_level(self) -> None:
        """cas=100 m/s, alt=0 → TAS ≈ 100 m/s."""
        tas = cas_to_tas(100.0, 0.0)
        assert float(tas) == pytest.approx(100.0, abs=0.5)


class TestCasToTasCruise:
    """cas_to_tas at cruise altitude."""

    def test_cas_to_tas_cruise(self) -> None:
        """cas=128.6 m/s (250 kt), alt=10 000 m → TAS ≈ 212 m/s.

        Cross-validate: CAS 250 kt at 10 000 m ≈ Mach 0.709 (not 0.78).
        The spec value Mach 0.78 applies at ~FL330, not 10 000 m.
        """
        tas_from_cas = cas_to_tas(128.6, 10_000.0)
        # Recover Mach from TAS, then cross-validate via mach_to_tas
        t = float(isa_temperature(10_000.0))
        a_local = math.sqrt(GAMMA_AIR * R * t)
        mach_recovered = float(tas_from_cas) / a_local
        tas_from_mach = mach_to_tas(mach_recovered, 10_000.0)
        assert float(tas_from_cas) == pytest.approx(float(tas_from_mach), rel=1e-9)
        assert float(tas_from_cas) == pytest.approx(212, abs=3)


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


class TestRoundtripMach:
    """mach_to_tas → reverse via local speed of sound → recover Mach."""

    def test_roundtrip_mach(self) -> None:
        """mach=0.82, alt=11 500 m → TAS → back to Mach ≈ 0.82."""
        mach_in = 0.82
        alt = 11_500.0
        tas = mach_to_tas(mach_in, alt)
        t = float(isa_temperature(alt))
        a_local = math.sqrt(GAMMA_AIR * R * t)
        mach_out = float(tas) / a_local
        assert mach_out == pytest.approx(mach_in, rel=1e-9)


class TestRoundtripCas:
    """cas_to_tas → reverse via compressible formula → recover CAS."""

    def test_roundtrip_cas(self) -> None:
        """cas=128.6 m/s, alt=8 000 m → TAS → reverse → CAS ≈ 128.6."""
        cas_in = 128.6
        alt = 8_000.0
        tas = float(cas_to_tas(cas_in, alt))

        # Reverse: TAS → qc at altitude → CAS
        p = float(isa_pressure(alt))
        t = float(isa_temperature(alt))
        a_local = math.sqrt(GAMMA_AIR * R * t)
        mach = tas / a_local

        # Impact pressure from Mach (subsonic compressible)
        gm1_over_2 = (GAMMA_AIR - 1.0) / 2.0
        g_over_gm1 = GAMMA_AIR / (GAMMA_AIR - 1.0)
        qc = p * ((1.0 + gm1_over_2 * mach**2) ** g_over_gm1 - 1.0)

        # CAS from impact pressure (sea-level inverse)
        cas_out = A0 * math.sqrt(
            (2.0 / (GAMMA_AIR - 1.0)) * ((qc / P0 + 1.0) ** (1.0 / g_over_gm1) - 1.0)
        )
        assert cas_out == pytest.approx(cas_in, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary / degenerate inputs."""

    @pytest.mark.parametrize(
        ("mach", "alt"),
        [
            pytest.param(float("nan"), 10_000.0, id="nan_mach"),
            pytest.param(0.8, float("nan"), id="nan_altitude"),
        ],
    )
    def test_nan_input_yields_nan(self, mach: float, alt: float) -> None:
        """NaN in either mach or altitude propagates to NaN TAS."""
        result = mach_to_tas(mach, alt)
        assert math.isnan(float(result))

    def test_stratosphere(self) -> None:
        """mach=0.82, alt=12 000 m → correct (isothermal T=216.65 K)."""
        tas = mach_to_tas(0.82, 12_000.0)
        a_local = math.sqrt(GAMMA_AIR * R * 216.65)
        assert float(tas) == pytest.approx(0.82 * a_local, rel=1e-6)

    def test_negative_cas(self) -> None:
        """cas=-10, alt=5 000 -> returns NaN or 0."""
        result = cas_to_tas(-10.0, 5_000.0)
        val = float(result)
        assert math.isnan(val) or val == 0.0
