"""Tests for node_fdm_data.conversions — unit conversion Polars expressions."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_series_equal

from node_fdm_data.conversions import (
    celsius_to_kelvin,
    deg_to_rad,
    ft_to_m,
    ftmin_to_ms,
    kelvin_to_celsius,
    kgh_to_kgs,
    kgs_to_kgh,
    kt_to_ms,
    m_to_ft,
    m_to_nm,
    ms_to_ftmin,
    ms_to_kt,
    nm_to_m,
    rad_to_deg,
)


class TestFtToM:
    """Feet ↔ metres conversions."""

    def test_ft_to_m(self) -> None:
        s = pl.Series("alt", [1000.0, 0.0, -100.0])
        result = pl.DataFrame({"alt": s}).select(ft_to_m("alt")).to_series()
        expected = pl.Series("alt", [304.8, 0.0, -30.48])
        assert_series_equal(result, expected)

    def test_m_to_ft_roundtrip(self) -> None:
        s = pl.Series("alt", [304.8])
        result = pl.DataFrame({"alt": s}).select(m_to_ft("alt")).to_series()
        assert result[0] == pytest.approx(1000.0)


class TestKtToMs:
    """Knots ↔ m/s conversions."""

    def test_kt_to_ms(self) -> None:
        s = pl.Series("spd", [100.0])
        result = pl.DataFrame({"spd": s}).select(kt_to_ms("spd")).to_series()
        assert result[0] == pytest.approx(51.4444, rel=1e-4)

    def test_ms_to_kt_roundtrip(self) -> None:
        s = pl.Series("spd", [51.4444])
        result = pl.DataFrame({"spd": s}).select(ms_to_kt("spd")).to_series()
        assert result[0] == pytest.approx(100.0, rel=1e-3)


class TestCelsiusToKelvin:
    """Temperature conversions."""

    def test_celsius_to_kelvin(self) -> None:
        s = pl.Series("temp", [0.0, -273.15])
        result = pl.DataFrame({"temp": s}).select(celsius_to_kelvin("temp")).to_series()
        expected = pl.Series("temp", [273.15, 0.0])
        assert_series_equal(result, expected)

    def test_kelvin_to_celsius_roundtrip(self) -> None:
        s = pl.Series("temp", [273.15])
        result = pl.DataFrame({"temp": s}).select(kelvin_to_celsius("temp")).to_series()
        assert result[0] == pytest.approx(0.0)


class TestFtminToMs:
    """ft/min ↔ m/s conversions."""

    def test_ftmin_to_ms(self) -> None:
        s = pl.Series("vz", [1000.0])
        result = pl.DataFrame({"vz": s}).select(ftmin_to_ms("vz")).to_series()
        assert result[0] == pytest.approx(0.3048 / 60.0 * 1000.0)

    def test_ms_to_ftmin_roundtrip(self) -> None:
        s = pl.Series("vz", [5.08])
        result = pl.DataFrame({"vz": s}).select(ms_to_ftmin("vz")).to_series()
        assert result[0] == pytest.approx(5.08 / (0.3048 / 60.0))


class TestNmToM:
    """Nautical miles ↔ metres."""

    def test_nm_to_m(self) -> None:
        s = pl.Series("d", [1.0])
        result = pl.DataFrame({"d": s}).select(nm_to_m("d")).to_series()
        assert result[0] == pytest.approx(1852.0)

    def test_m_to_nm_roundtrip(self) -> None:
        s = pl.Series("d", [1852.0])
        result = pl.DataFrame({"d": s}).select(m_to_nm("d")).to_series()
        assert result[0] == pytest.approx(1.0)


class TestDegToRad:
    """Degree ↔ radian conversions."""

    def test_deg_to_rad(self) -> None:
        s = pl.Series("a", [180.0])
        result = pl.DataFrame({"a": s}).select(deg_to_rad("a")).to_series()
        assert result[0] == pytest.approx(math.pi)

    def test_rad_to_deg_roundtrip(self) -> None:
        s = pl.Series("a", [math.pi])
        result = pl.DataFrame({"a": s}).select(rad_to_deg("a")).to_series()
        assert result[0] == pytest.approx(180.0)


class TestKghToKgs:
    """kg/h ↔ kg/s conversions."""

    def test_kgh_to_kgs(self) -> None:
        s = pl.Series("ff", [3600.0])
        result = pl.DataFrame({"ff": s}).select(kgh_to_kgs("ff")).to_series()
        assert result[0] == pytest.approx(1.0)

    def test_kgs_to_kgh_roundtrip(self) -> None:
        s = pl.Series("ff", [1.0])
        result = pl.DataFrame({"ff": s}).select(kgs_to_kgh("ff")).to_series()
        assert result[0] == pytest.approx(3600.0)


class TestEdgeCases:
    """Edge cases from ticket spec."""

    def test_nan_in_conversion(self) -> None:
        """NaN in conversion → NaN propagated (Polars default)."""
        s = pl.Series("alt", [1000.0, None, -100.0])
        result = pl.DataFrame({"alt": s}).select(ft_to_m("alt")).to_series()
        assert result[0] == pytest.approx(304.8)
        assert result[1] is None
        assert result[2] == pytest.approx(-30.48)
