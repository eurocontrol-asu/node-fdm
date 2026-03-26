"""Tests for the EKF + RTS smoother module."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from node_fdm_data.preprocessing.kalman import (
    extended_kalman_filter,
    rts_smoother,
    smooth_flight,
    smooth_flights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timestamps_s(n: int, dt: float = 4.0) -> np.ndarray:
    """Create regular timestamps in seconds."""
    return np.arange(n, dtype=np.float64) * dt


def _make_flight_df(  # noqa: PLR0913
    n: int = 100,
    *,
    tas_kt: float = 450.0,
    gamma_rad: float = 0.0,
    alt_ft: float = 35000.0,
    noise_tas: float = 0.0,
    noise_gamma: float = 0.0,
    noise_alt: float = 0.0,
    dt_s: int = 4,
) -> pl.DataFrame:
    """Build a synthetic single-flight DataFrame."""
    rng = np.random.default_rng(42)
    t0 = datetime(2025, 1, 1)
    timestamps = [t0 + timedelta(seconds=i * dt_s) for i in range(n)]

    return pl.DataFrame(
        {
            "raw_timestamp": timestamps,
            "ekf_input_tas_kt": [tas_kt + rng.normal(0, noise_tas) for _ in range(n)],
            "fdm_gamma_rad": [gamma_rad + rng.normal(0, noise_gamma) for _ in range(n)],
            "raw_alt_ft": [alt_ft + rng.normal(0, noise_alt) for _ in range(n)],
            "meta_flight_id": ["flight_01"] * n,
        }
    )


# ---------------------------------------------------------------------------
# EKF core tests
# ---------------------------------------------------------------------------


class TestEKFConstantState:
    """Constant TAS + altitude → output should match input closely."""

    def test_constant_tas_alt(self) -> None:
        n = 50
        ts = _make_timestamps_s(n)
        tas_ms = np.full(n, 230.0)
        gamma = np.zeros(n)
        alt_m = np.full(n, 10668.0)
        measurements = np.column_stack([tas_ms, gamma, alt_m])

        x0 = np.array([230.0, 0.0, 10668.0, 0.0, 0.0])
        p0 = np.eye(5) * 1e4
        r_noise = np.diag([1.0, 0.001, 10.0])
        q_noise = np.diag([0.1, 0.01, 0.01, 0.5, 0.1])

        states, _ = extended_kalman_filter(
            measurements, ts, x0, p0, q_noise, r_noise, reject_sigma=3.0
        )

        # After convergence, TAS should be close to 230
        assert states[-1, 0] == pytest.approx(230.0, abs=2.0)
        # Altitude should be close to 10668
        assert states[-1, 2] == pytest.approx(10668.0, abs=50.0)
        # Derivatives should be near 0
        assert abs(states[-1, 3]) < 1.0
        assert abs(states[-1, 4]) < 0.1


class TestEKFNoisyClimb:
    """Noisy climb -- output should be smoother (lower std)."""

    def test_smoother_output(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        ts = _make_timestamps_s(n)

        # Constant TAS with noise, climbing (positive gamma)
        tas_true = np.full(n, 230.0)
        gamma_true = np.full(n, 0.05)  # ~3° climb
        alt_true = 10000.0 + np.cumsum(tas_true * np.sin(gamma_true) * 4.0)

        noise_tas = rng.normal(0, 3.0, n)
        noise_gamma = rng.normal(0, 0.01, n)
        noise_alt = rng.normal(0, 30.0, n)

        measurements = np.column_stack(
            [tas_true + noise_tas, gamma_true + noise_gamma, alt_true + noise_alt]
        )

        x0 = np.array([measurements[0, 0], measurements[0, 1], measurements[0, 2], 0.0, 0.0])
        p0 = np.eye(5) * 1e4
        r_noise = np.diag([9.0, 0.0001, 900.0])
        q_noise = np.diag([0.1, 0.01, 0.01, 0.5, 0.1])

        states, covs = extended_kalman_filter(
            measurements, ts, x0, p0, q_noise, r_noise, reject_sigma=3.0
        )
        smoothed = rts_smoother(states, covs, q_noise, ts)

        # Smoothed TAS should have lower std than raw
        raw_std = np.std(measurements[10:, 0])
        smooth_std = np.std(smoothed[10:, 0])
        assert smooth_std < raw_std


class TestEKFOutlierRejection:
    """Spike injection → EKF should reject it."""

    def test_spike_rejected(self) -> None:
        n = 50
        ts = _make_timestamps_s(n)
        tas_ms = np.full(n, 230.0)
        gamma = np.zeros(n)
        alt_m = np.full(n, 10668.0)

        # Inject a big spike at index 25
        tas_ms_noisy = tas_ms.copy()
        tas_ms_noisy[25] = 500.0  # way off

        measurements = np.column_stack([tas_ms_noisy, gamma, alt_m])

        x0 = np.array([230.0, 0.0, 10668.0, 0.0, 0.0])
        p0 = np.eye(5) * 1e4
        r_noise = np.diag([1.0, 0.001, 10.0])
        q_noise = np.diag([0.1, 0.01, 0.01, 0.5, 0.1])

        states, _ = extended_kalman_filter(
            measurements, ts, x0, p0, q_noise, r_noise, reject_sigma=3.0
        )

        # State at index 25 should NOT follow the spike
        assert states[25, 0] == pytest.approx(230.0, abs=10.0)
        # State after spike should recover
        assert states[26, 0] == pytest.approx(230.0, abs=5.0)


class TestEKFNaNHandling:
    """NaN measurements should be replaced by prediction."""

    def test_nan_measurements(self) -> None:
        n = 30
        ts = _make_timestamps_s(n)
        tas_ms = np.full(n, 230.0)
        gamma = np.zeros(n)
        alt_m = np.full(n, 10668.0)
        measurements = np.column_stack([tas_ms, gamma, alt_m])

        # Insert NaNs
        measurements[10:15, 0] = np.nan  # TAS gap
        measurements[20, :] = np.nan  # Full row gap

        x0 = np.array([230.0, 0.0, 10668.0, 0.0, 0.0])
        p0 = np.eye(5) * 1e4
        r_noise = np.diag([1.0, 0.001, 10.0])
        q_noise = np.diag([0.1, 0.01, 0.01, 0.5, 0.1])

        states, _ = extended_kalman_filter(
            measurements, ts, x0, p0, q_noise, r_noise, reject_sigma=3.0
        )

        # States should have no NaN
        assert not np.any(np.isnan(states))
        # After NaN gap, should still be close to truth
        assert states[15, 0] == pytest.approx(230.0, abs=5.0)


# ---------------------------------------------------------------------------
# RTS smoother tests
# ---------------------------------------------------------------------------


class TestRTSSmoother:
    """RTS backward pass should reduce error."""

    def test_reduces_error(self) -> None:
        n = 60
        rng = np.random.default_rng(42)
        ts = _make_timestamps_s(n)

        tas_true = np.full(n, 230.0)
        gamma_true = np.zeros(n)
        alt_true = np.full(n, 10668.0)

        noise = rng.normal(0, 3.0, n)
        measurements = np.column_stack([tas_true + noise, gamma_true, alt_true])

        x0 = np.array([230.0, 0.0, 10668.0, 0.0, 0.0])
        p0 = np.eye(5) * 1e4
        r_noise = np.diag([9.0, 0.001, 10.0])
        q_noise = np.diag([0.1, 0.01, 0.01, 0.5, 0.1])

        states, covs = extended_kalman_filter(measurements, ts, x0, p0, q_noise, r_noise)
        smoothed = rts_smoother(states, covs, q_noise, ts)

        # Early points should benefit most from smoothing
        fwd_err = abs(states[5, 0] - 230.0)
        smooth_err = abs(smoothed[5, 0] - 230.0)
        assert smooth_err <= fwd_err + 0.5  # smoother should not be worse


# ---------------------------------------------------------------------------
# Polars integration tests
# ---------------------------------------------------------------------------


class TestSmoothFlight:
    """Integration test with Polars DataFrame."""

    def test_output_columns_exist(self) -> None:
        df = _make_flight_df(50, noise_tas=2.0, noise_gamma=0.005, noise_alt=20.0)
        result = smooth_flight(df)

        for col in ["ekf_tas_ms", "ekf_gamma_rad", "ekf_alt_m", "ekf_dtas_ms2", "ekf_dgamma_rads"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_finite(self) -> None:
        df = _make_flight_df(50, noise_tas=2.0, noise_gamma=0.005, noise_alt=20.0)
        result = smooth_flight(df)

        assert result["ekf_tas_ms"].null_count() == 0
        assert result["ekf_gamma_rad"].null_count() == 0
        assert result["ekf_alt_m"].null_count() == 0

    def test_short_flight_returns_nulls(self) -> None:
        """Flights with < 3 points return null columns."""
        df = _make_flight_df(2)
        result = smooth_flight(df)

        assert "ekf_tas_ms" in result.columns
        assert result["ekf_tas_ms"].null_count() == 2


class TestSmoothFlights:
    """Batch processing over multiple flights."""

    def test_preserves_flight_count(self) -> None:
        df1 = _make_flight_df(50, noise_tas=2.0).with_columns(pl.lit("f1").alias("meta_flight_id"))
        df2 = _make_flight_df(60, noise_tas=3.0).with_columns(pl.lit("f2").alias("meta_flight_id"))
        df = pl.concat([df1, df2])

        result = smooth_flights(df)
        assert result["meta_flight_id"].n_unique() == 2
        assert len(result) == len(df)

    def test_idempotent(self) -> None:
        """Running twice gives same result."""
        df = _make_flight_df(50, noise_tas=2.0)
        result1 = smooth_flights(df)
        result2 = smooth_flights(result1)

        np.testing.assert_array_almost_equal(
            result2["ekf_tas_ms"].to_numpy(),
            result1["ekf_tas_ms"].to_numpy(),
        )
