"""Unit tests for `visualize` private helpers (no I/O, no matplotlib render)."""

from __future__ import annotations

import numpy as np


def test_integrate_ground_track_zero_wind_constant_heading() -> None:
    """With zero wind, constant heading north (ψ=0), and constant TAS, the
    integrated track moves due north at the expected per-step rate."""
    from node_fdm_pipeline.commands.visualize import _integrate_ground_track

    n = 5
    tas = np.full(n, 100.0)  # m/s
    gamma = np.zeros(n)
    heading = np.zeros(n)  # north
    u_wind = np.zeros(n)
    v_wind = np.zeros(n)

    lat_pred, lon_pred = _integrate_ground_track(
        lat0=0.0,
        lon0=0.0,
        heading_pred=heading,
        tas_pred=tas,
        gamma_pred=gamma,
        u_wind=u_wind,
        v_wind=v_wind,
        step_s=4.0,
    )

    # Per step: 100 m/s x 4 s = 400 m → 400 / R_earth x 180/π ≈ 0.003595°
    expected_step = np.degrees(400.0 / 6_371_000.0)
    assert lat_pred[0] == 0.0
    assert lon_pred[0] == 0.0
    np.testing.assert_allclose(lat_pred[1:], expected_step * np.arange(1, n), atol=1e-6)
    # Lon stays at 0 (no east component)
    np.testing.assert_allclose(lon_pred, 0.0, atol=1e-9)


def test_integrate_ground_track_pure_east_wind_drifts_east() -> None:
    """Heading north, pure east wind → ground track drifts east while still
    advancing north at the TAS rate."""
    from node_fdm_pipeline.commands.visualize import _integrate_ground_track

    n = 3
    tas = np.full(n, 100.0)
    gamma = np.zeros(n)
    heading = np.zeros(n)
    u_wind = np.full(n, 50.0)  # m/s east
    v_wind = np.zeros(n)

    lat_pred, lon_pred = _integrate_ground_track(
        lat0=0.0,
        lon0=0.0,
        heading_pred=heading,
        tas_pred=tas,
        gamma_pred=gamma,
        u_wind=u_wind,
        v_wind=v_wind,
        step_s=4.0,
    )

    # First step east: 50 m/s x 4 s = 200 m east at lat=0 → 200/Rx180/π
    expected_lon_step = np.degrees(200.0 / 6_371_000.0)
    np.testing.assert_allclose(lon_pred[1], expected_lon_step, atol=1e-6)
    # Lat advances at TAS rate as before
    expected_lat_step = np.degrees(400.0 / 6_371_000.0)
    np.testing.assert_allclose(lat_pred[1], expected_lat_step, atol=1e-6)
