"""Unit tests for heading wrap-around and lateral position metrics (AXM-1685)."""

from __future__ import annotations

import math

import polars as pl
import pytest

from node_fdm_pipeline.commands.evaluate import (
    compute_errors_by_phase,
    compute_position_errors_by_phase,
)


def test_compute_errors_by_phase_heading_wraps_around_360() -> None:
    """AC1: pred=359 deg, true=1 deg -> MAE approx 2 deg (not 358 deg)."""
    df = pl.DataFrame(
        {
            "pred_fdm_heading_rad": [math.radians(359.0)] * 4,
            "fdm_heading_rad": [math.radians(1.0)] * 4,
            "raw_vz_ms": [0.0] * 4,
        }
    )
    result = compute_errors_by_phase(
        df,
        pred_col="pred_fdm_heading_rad",
        target_col="fdm_heading_rad",
        vertical_rate_col="raw_vz_ms",
    )
    mae_all = result.filter(pl.col("Phase") == "All phases")["MAE"][0]
    assert mae_all == pytest.approx(2.0, abs=1e-6)


def test_compute_errors_by_phase_heading_mape_is_nan() -> None:
    """AC1: MAPE for heading is NaN (angles are not amenable to ratio errors)."""
    df = pl.DataFrame(
        {
            "pred_fdm_heading_rad": [math.radians(10.0), math.radians(20.0)],
            "fdm_heading_rad": [math.radians(15.0), math.radians(25.0)],
            "raw_vz_ms": [0.0, 0.0],
        }
    )
    result = compute_errors_by_phase(
        df,
        pred_col="pred_fdm_heading_rad",
        target_col="fdm_heading_rad",
        vertical_rate_col="raw_vz_ms",
    )
    mape_values = result["MAPE (%)"].to_list()
    assert mape_values, "expected at least one row of metrics"
    assert all(math.isnan(v) for v in mape_values)


def test_compute_position_errors_returns_haversine_distance() -> None:
    """AC3: two points 1 deg apart in latitude -> ~111_195 m haversine distance."""
    df = pl.DataFrame(
        {
            "pred_lat_deg": [0.0, 0.0, 0.0],
            "pred_lon_deg": [0.0, 0.0, 0.0],
            "lat_deg": [1.0, 1.0, 1.0],
            "lon_deg": [0.0, 0.0, 0.0],
            "raw_vz_ms": [0.0, 0.0, 0.0],
        }
    )
    result = compute_position_errors_by_phase(
        df,
        lat_pred_col="pred_lat_deg",
        lon_pred_col="pred_lon_deg",
        lat_true_col="lat_deg",
        lon_true_col="lon_deg",
        vertical_rate_col="raw_vz_ms",
    )
    assert result is not None
    mae_all = result.filter(pl.col("Phase") == "All phases")["MAE"][0]
    assert mae_all == pytest.approx(111_195.0, abs=50.0)


def test_compute_position_errors_skips_when_lat_lon_missing() -> None:
    """AC6: when lat/lon columns are missing, function returns None (skip pattern)."""
    df = pl.DataFrame({"raw_vz_ms": [0.0, 0.0]})
    result = compute_position_errors_by_phase(
        df,
        lat_pred_col="pred_lat_deg",
        lon_pred_col="pred_lon_deg",
        lat_true_col="lat_deg",
        lon_true_col="lon_deg",
        vertical_rate_col="raw_vz_ms",
    )
    assert result is None or len(result) == 0
