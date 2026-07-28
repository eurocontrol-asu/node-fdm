"""Unit tests for ``node_fdm_data.physics.wmm``.

``declination_vec`` is a vectorized port of :meth:`pygeomag.GeoMag.calculate`
that documents a *bit-exact* contract with the scalar reference.  That upstream
implementation is therefore the oracle used throughout this module: every
assertion compares against ``GeoMag().calculate(...).d`` rather than against
hard-coded expectations, so the tests stay valid across WMM coefficient
epochs while still failing on any drift in the harmonic synthesis itself.

The looser "known location ± 1°" checks live in ``test_magnetic.py``, which
covers the public wrapper; here the tolerance is exact equality.
"""

from __future__ import annotations

import numpy as np
import pytest
from pygeomag import GeoMag  # type: ignore[import-untyped]

from node_fdm_data.physics.wmm import declination_vec

_DECIMAL_YEAR = 2025.5


def _reference(glat: float, glon: float, alt_km: float, year: float = _DECIMAL_YEAR) -> float:
    """Scalar declination from pygeomag — the oracle ``declination_vec`` ports."""
    return float(GeoMag().calculate(glat=glat, glon=glon, alt=alt_km, time=year).d)


@pytest.fixture
def geo() -> GeoMag:
    return GeoMag()


@pytest.mark.parametrize(
    ("glat", "glon", "alt_km"),
    [
        (48.85, 2.35, 0.0),  # Paris, sea level
        (40.70, -74.00, 10.0),  # New York, cruise altitude
        (1.35, 103.80, 5.0),  # Singapore, near the magnetic equator
        (-33.90, 151.20, 0.0),  # Sydney, southern hemisphere
        (70.00, -40.00, 0.0),  # Greenland, high declination
        (0.0, 0.0, 0.0),  # null island — origin of both angle sweeps
    ],
)
def test_matches_pygeomag_scalar_reference(
    geo: GeoMag, glat: float, glon: float, alt_km: float
) -> None:
    """Each sample reproduces the scalar reference bit-for-bit."""
    result = declination_vec(
        geo, np.array([glat]), np.array([glon]), np.array([alt_km]), _DECIMAL_YEAR
    )
    assert result[0] == _reference(glat, glon, alt_km)


def test_batch_matches_pointwise_reference(geo: GeoMag) -> None:
    """A batched call equals the per-sample reference for every element.

    Guards the vectorization itself: a broadcasting or indexing error would
    let neighbouring samples leak into one another.
    """
    lat = np.array([48.85, 40.70, 1.35, -33.90, 70.00])
    lon = np.array([2.35, -74.00, 103.80, 151.20, -40.00])
    alt = np.array([0.0, 10.0, 5.0, 0.0, 0.0])

    result = declination_vec(geo, lat, lon, alt, _DECIMAL_YEAR)

    expected = [_reference(la, lo, al) for la, lo, al in zip(lat, lon, alt, strict=True)]
    assert result.tolist() == expected


def test_batching_does_not_change_results(geo: GeoMag) -> None:
    """Evaluating a point alone or inside a batch yields the same value."""
    lat = np.array([48.85, 40.70, 1.35])
    lon = np.array([2.35, -74.00, 103.80])
    alt = np.array([0.0, 10.0, 5.0])

    batched = declination_vec(geo, lat, lon, alt, _DECIMAL_YEAR)
    individually = np.concatenate(
        [
            declination_vec(geo, lat[i : i + 1], lon[i : i + 1], alt[i : i + 1], _DECIMAL_YEAR)
            for i in range(lat.size)
        ]
    )
    assert batched.tolist() == individually.tolist()


@pytest.mark.parametrize("glat", [90.0, -90.0])
def test_geographic_poles_use_bpp_branch(geo: GeoMag, glat: float) -> None:
    """At the poles ``st == 0`` selects the ``bpp`` fallback, not a divide-by-zero.

    ``_assemble_declination`` divides ``bp`` by ``st``; exactly at the poles
    that denominator vanishes and the code must substitute ``bpp``.  A
    regression here surfaces as NaN/inf rather than a finite angle.
    """
    result = declination_vec(
        geo, np.array([glat]), np.array([0.0]), np.array([0.0]), _DECIMAL_YEAR
    )

    assert np.isfinite(result).all()
    assert result[0] == _reference(glat, 0.0, 0.0)


def test_altitude_changes_declination(geo: GeoMag) -> None:
    """Altitude is a real input to the synthesis, not a silently ignored argument."""
    lat = np.array([70.0])
    lon = np.array([-40.0])

    sea_level = declination_vec(geo, lat, lon, np.array([0.0]), _DECIMAL_YEAR)
    high = declination_vec(geo, lat, lon, np.array([100.0]), _DECIMAL_YEAR)

    assert sea_level[0] != high[0]
    assert high[0] == _reference(70.0, -40.0, 100.0)


def test_secular_variation_applied(geo: GeoMag) -> None:
    """``decimal_year`` drives the secular-variation term ``dt * cd``."""
    lat = np.array([48.85])
    lon = np.array([2.35])
    alt = np.array([0.0])

    # ``_epoch`` is populated lazily by ``_load_coefficients``, which
    # ``declination_vec`` calls internally — read it only after a warm-up.
    at_epoch = declination_vec(geo, lat, lon, alt, _DECIMAL_YEAR)
    assert geo._epoch is not None
    epoch = float(geo._epoch)
    later = declination_vec(geo, lat, lon, alt, epoch + 4.0)

    assert at_epoch[0] != later[0]
    assert later[0] == _reference(48.85, 2.35, 0.0, epoch + 4.0)


def test_longitude_wrap_is_consistent(geo: GeoMag) -> None:
    """Longitudes 180° apart in representation (-170 vs 190) agree.

    Exercises the recursive sin/cos longitude harmonics, which must be
    periodic in 360°.
    """
    lat = np.array([45.0, 45.0])
    lon = np.array([-170.0, 190.0])
    alt = np.array([0.0, 0.0])

    result = declination_vec(geo, lat, lon, alt, _DECIMAL_YEAR)

    assert result[0] == pytest.approx(result[1], abs=1e-9)


def test_output_shape_and_dtype(geo: GeoMag) -> None:
    """Output is a float64 array of the same length as the inputs."""
    n = 25
    lat = np.linspace(-60.0, 60.0, n)
    lon = np.linspace(-180.0, 180.0, n)
    alt = np.full(n, 11.0)

    result = declination_vec(geo, lat, lon, alt, _DECIMAL_YEAR)

    assert result.shape == (n,)
    assert result.dtype == np.float64
    assert np.isfinite(result).all()


def test_declination_within_valid_angular_range(geo: GeoMag) -> None:
    """Declination is an ``arctan2`` output, so it stays within (-180, 180]."""
    lat = np.linspace(-89.0, 89.0, 40)
    lon = np.linspace(-179.0, 179.0, 40)
    alt = np.zeros(40)

    result = declination_vec(geo, lat, lon, alt, _DECIMAL_YEAR)

    assert np.all(result > -180.0)
    assert np.all(result <= 180.0)


def test_accepts_non_array_sequences(geo: GeoMag) -> None:
    """Inputs go through ``np.asarray``, so plain lists are valid."""
    result = declination_vec(geo, [48.85], [2.35], [0.0], _DECIMAL_YEAR)  # type: ignore[arg-type]

    assert result.shape == (1,)
    assert result[0] == _reference(48.85, 2.35, 0.0)
