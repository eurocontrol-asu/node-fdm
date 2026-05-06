"""Unit tests for ``node_fdm_data.physics.magnetic``."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from node_fdm_data.physics.magnetic import magnetic_declination

_TS_2025 = _dt.datetime(2025, 6, 1, tzinfo=_dt.UTC)


@pytest.mark.parametrize(
    ("lat", "lon", "expected", "tol"),
    [
        # Paris (48.85N, 2.35E): WMM-2025 ~ +1.5° E.
        (48.85, 2.35, 1.5, 1.0),
        # New York (40.7N, 74.0W): WMM-2025 ~ -13° W.
        (40.7, -74.0, -13.0, 1.5),
        # Singapore (1.35N, 103.8E): WMM-2025 ~ +0.2° E.
        (1.35, 103.8, 0.2, 1.0),
    ],
)
def test_magnetic_declination_known_locations(
    lat: float, lon: float, expected: float, tol: float
) -> None:
    result = magnetic_declination(
        np.array([lat]),
        np.array([lon]),
        np.array([0.0]),
        _TS_2025,
    )
    assert result.shape == (1,)
    assert abs(float(result[0]) - expected) < tol, (
        f"declination at ({lat}, {lon}) = {result[0]:.3f}, expected ~{expected} ± {tol}"
    )


def test_magnetic_declination_array_shape_preserved() -> None:
    n = 50
    lat = np.linspace(40.0, 50.0, n)
    lon = np.linspace(-5.0, 5.0, n)
    alt = np.full(n, 35_000.0)
    result = magnetic_declination(lat, lon, alt, _TS_2025)
    assert result.shape == (n,)
    assert np.isfinite(result).all()


def test_magnetic_declination_nan_inputs() -> None:
    lat = np.array([48.85, np.nan, 1.35])
    lon = np.array([2.35, 0.0, np.nan])
    alt = np.array([0.0, 0.0, 0.0])
    result = magnetic_declination(lat, lon, alt, _TS_2025)
    assert np.isfinite(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_magnetic_declination_naive_timestamp_accepted() -> None:
    naive = _dt.datetime(2025, 6, 1)  # naive datetime intentionally tested
    result = magnetic_declination(
        np.array([48.85]),
        np.array([2.35]),
        np.array([0.0]),
        naive,
    )
    assert np.isfinite(result[0])
