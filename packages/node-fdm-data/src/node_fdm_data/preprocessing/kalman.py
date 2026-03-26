"""Extended Kalman Filter + RTS smoother for flight dynamics parameters.

Adapted from the ``traffic`` library's ``algorithms/filters/ekf.py``.
Smooths TAS, flight-path angle (gamma), and altitude using a non-linear
state-transition model with analytical Jacobian and per-component
outlier gating.

State vector (5D):
    [TAS (m/s), gamma (rad), h (m), dTAS/dt (m/s2), dgamma/dt (rad/s)]

Measurements (3D):
    [TAS, gamma, h] -- derivatives are hidden states estimated by the filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    import polars as pl

import numpy.typing as npt

__all__ = [
    "extended_kalman_filter",
    "rts_smoother",
    "smooth_flight",
    "smooth_flights",
]

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
_KT_TO_MS: float = 0.514444
_FT_TO_M: float = 0.3048
_FTMIN_TO_MS: float = _FT_TO_M / 60.0

# Index aliases for readability
_TAS, _GAMMA, _H, _DTAS, _DGAMMA = 0, 1, 2, 3, 4
_N_STATES = 5
_N_MEAS = 3  # we observe TAS, gamma, h
_MIN_POINTS = 3  # minimum measurements for the filter


# ---------------------------------------------------------------------------
# Non-linear state transition
# ---------------------------------------------------------------------------


def _state_transition(x: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """Predict next state from current state.

    Model::

        TAS_next  = TAS + dTAS * dt
        gamma_next = gamma + dgamma * dt
        h_next    = h + TAS * sin(gamma) * dt
        dTAS_next = dTAS   (constant acceleration)
        dgamma_next = dgamma  (constant rate)
    """
    x_pred = x.copy()
    x_pred[_TAS] += x[_DTAS] * dt
    x_pred[_GAMMA] += x[_DGAMMA] * dt
    x_pred[_H] += x[_TAS] * np.sin(x[_GAMMA]) * dt
    return x_pred


def _jacobian(
    x: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Analytical Jacobian of the state transition.

    Returns the matrix ``F = df/dx`` evaluated at current state.
    """
    f_jac = np.eye(_N_STATES)
    # dTAS_next / d(dTAS)
    f_jac[_TAS, _DTAS] = dt
    # dgamma_next / d(dgamma)
    f_jac[_GAMMA, _DGAMMA] = dt
    # dh_next / dTAS = sin(gamma) * dt
    f_jac[_H, _TAS] = np.sin(x[_GAMMA]) * dt
    # dh_next / dgamma = TAS * cos(gamma) * dt
    f_jac[_H, _GAMMA] = x[_TAS] * np.cos(x[_GAMMA]) * dt
    return f_jac


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

# Constant: we observe the first 3 states directly
_H_OBS = np.zeros((_N_MEAS, _N_STATES))
_H_OBS[0, _TAS] = 1.0
_H_OBS[1, _GAMMA] = 1.0
_H_OBS[2, _H] = 1.0


# ---------------------------------------------------------------------------
# EKF core
# ---------------------------------------------------------------------------


def extended_kalman_filter(  # noqa: PLR0913
    measurements: npt.NDArray[np.float64],
    timestamps_s: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
    q_noise: npt.NDArray[np.float64],
    r_noise: npt.NDArray[np.float64],
    reject_sigma: float = 3.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the Extended Kalman Filter.

    Args:
        measurements: ``(N, 3)`` array of ``[TAS, gamma, h]`` observations.
        timestamps_s: ``(N,)`` array of timestamps in seconds (monotonic).
        x0: Initial state ``(5,)``.
        p0: Initial covariance ``(5, 5)``.
        q_noise: Process noise ``(5, 5)``.
        r_noise: Measurement noise ``(3, 3)``.
        reject_sigma: Per-component gating threshold (sigma).

    Returns:
        Tuple of ``(states, covariances)`` -- each ``(N, 5)`` and
        ``(N, 5, 5)`` respectively.
    """
    n_steps = len(measurements)
    states = np.zeros((n_steps, _N_STATES))
    covariances = np.zeros((n_steps, _N_STATES, _N_STATES))

    states[0] = x0
    covariances[0] = p0

    x = x0.copy()
    p_cov = p0.copy()

    for i in range(1, n_steps):
        dt = timestamps_s[i] - timestamps_s[i - 1]
        if dt <= 0:
            # Duplicate timestamp -- carry forward
            states[i] = x
            covariances[i] = p_cov
            continue

        # --- Prediction ---
        f_jac = _jacobian(x, dt)
        x_pred = _state_transition(x, dt)
        p_pred = f_jac @ p_cov @ f_jac.T + q_noise

        # --- Measurement update ---
        z = measurements[i]
        h_mat = _H_OBS.copy()

        # Innovation
        nu = z - h_mat @ x_pred
        s_inn = h_mat @ p_pred @ h_mat.T + r_noise
        std_devs = np.sqrt(np.diag(s_inn))

        # Per-component gating (like traffic's EKF)
        for j in range(_N_MEAS):
            if np.isnan(nu[j]) or np.isnan(std_devs[j]):
                # Missing measurement -- use prediction
                z[j] = (h_mat @ x_pred)[j]
                h_mat[j, :] = 0.0
            elif abs(nu[j]) > reject_sigma * std_devs[j]:
                # Outlier -- replace with prediction
                z[j] = (h_mat @ x_pred)[j]
                h_mat[j, :] = 0.0

        # Recompute innovation after gating
        nu = z - h_mat @ x_pred
        s_inn = h_mat @ p_pred @ h_mat.T + r_noise

        # Kalman gain via solve (numerically stable)
        k_gain = linalg.solve(s_inn, h_mat @ p_pred, assume_a="pos").T

        # State update
        x = x_pred + k_gain @ nu
        # Covariance update
        p_cov = (np.eye(_N_STATES) - k_gain @ h_mat) @ p_pred

        states[i] = x
        covariances[i] = p_cov

    return states, covariances


# ---------------------------------------------------------------------------
# RTS Smoother
# ---------------------------------------------------------------------------


def rts_smoother(
    states: npt.NDArray[np.float64],
    covariances: npt.NDArray[np.float64],
    q_noise: npt.NDArray[np.float64],
    timestamps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Rauch-Tung-Striebel backward smoother.

    Args:
        states: ``(N, 5)`` forward-filtered states.
        covariances: ``(N, 5, 5)`` forward-filtered covariances.
        q_noise: Process noise ``(5, 5)``.
        timestamps_s: ``(N,)`` timestamps in seconds.

    Returns:
        ``(N, 5)`` smoothed states.
    """
    n = len(states)
    smoothed = states.copy()
    smoothed_cov = covariances.copy()

    for i in range(n - 2, -1, -1):
        dt = timestamps_s[i + 1] - timestamps_s[i]
        if dt <= 0:
            continue

        f_jac = _jacobian(states[i], dt)
        x_pred = _state_transition(states[i], dt)
        p_pred = f_jac @ covariances[i] @ f_jac.T + q_noise

        # Smoother gain
        g_gain = covariances[i] @ f_jac.T @ np.linalg.inv(p_pred)

        smoothed[i] = states[i] + g_gain @ (smoothed[i + 1] - x_pred)
        smoothed_cov[i] = covariances[i] + g_gain @ (smoothed_cov[i + 1] - p_pred) @ g_gain.T

    return smoothed


# ---------------------------------------------------------------------------
# High-level wrappers (Polars integration)
# ---------------------------------------------------------------------------


def _estimate_noise_matrix(
    arr: npt.NDArray[np.float64],
    window: int = 17,
) -> npt.NDArray[np.float64]:
    """Estimate diagonal measurement noise from rolling-window residuals.

    Matches the traffic convention: sigma2 = std(signal - rolling_mean)^2.
    """
    import pandas as pd

    n_cols = arr.shape[1]
    sigmas = np.zeros(n_cols)
    for j in range(n_cols):
        series = pd.Series(arr[:, j])
        residual = series - series.rolling(window, center=True, min_periods=1).mean()
        sigma = residual.std()
        sigmas[j] = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
    return np.diag(sigmas**2)


def smooth_flight(
    df: pl.DataFrame,
    *,
    reject_sigma: float = 3.0,
    rolling_window: int = 17,
) -> pl.DataFrame:
    """Apply EKF + RTS smoother to a single flight.

    Reads ``ekf_input_tas_kt``, ``fdm_gamma_rad``, ``raw_alt_ft`` and
    ``raw_timestamp``.  Writes ``ekf_tas_ms``, ``ekf_gamma_rad``,
    ``ekf_alt_m``, ``ekf_dtas_ms2``, ``ekf_dgamma_rads``.

    Args:
        df: Single-flight DataFrame (already merged via
            :func:`~node_fdm_data.preprocessing.merge.merge_bds_era5`).
        reject_sigma: Per-component outlier gating threshold.
        rolling_window: Window size for empirical R estimation.

    Returns:
        DataFrame with ``ekf_*`` output columns added.
    """
    import polars as pl

    # --- Extract & convert to SI ---
    tas_kt = df["ekf_input_tas_kt"].to_numpy().astype(np.float64)
    gamma_rad = df["fdm_gamma_rad"].to_numpy().astype(np.float64)
    alt_ft = df["raw_alt_ft"].to_numpy().astype(np.float64)

    tas_ms = tas_kt * _KT_TO_MS
    alt_m = alt_ft * _FT_TO_M

    # Timestamps in seconds
    ts = df["raw_timestamp"].cast(pl.Int64).to_numpy().astype(np.float64) / 1e6
    ts = ts - ts[0]  # relative

    # Guard: need at least 3 points for the filter
    if len(ts) < _MIN_POINTS:
        return df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("ekf_tas_ms"),
            pl.lit(None).cast(pl.Float64).alias("ekf_gamma_rad"),
            pl.lit(None).cast(pl.Float64).alias("ekf_alt_m"),
            pl.lit(None).cast(pl.Float64).alias("ekf_dtas_ms2"),
            pl.lit(None).cast(pl.Float64).alias("ekf_dgamma_rads"),
        )

    # --- Build measurement matrix ---
    measurements = np.column_stack([tas_ms, gamma_rad, alt_m])

    # --- Initial state (derivatives start at 0) ---
    # Use first valid (non-NaN) values for initial state
    x0 = np.zeros(_N_STATES)
    for j, col in enumerate([tas_ms, gamma_rad, alt_m]):
        finite = col[np.isfinite(col)]
        x0[j] = finite[0] if len(finite) > 0 else 0.0

    p0 = np.eye(_N_STATES) * 1e4

    # --- Noise matrices ---
    r_noise = _estimate_noise_matrix(measurements, window=rolling_window)
    q_noise = np.diag([0.1, 0.3, 0.01, 1.0, 0.5]) * np.mean(np.diag(r_noise))

    # --- Run EKF + RTS ---
    states, covariances = extended_kalman_filter(
        measurements, ts, x0, p0, q_noise, r_noise, reject_sigma=reject_sigma
    )
    smoothed = rts_smoother(states, covariances, q_noise, ts)

    # --- Write output columns ---
    return df.with_columns(
        pl.Series("ekf_tas_ms", smoothed[:, _TAS]),
        pl.Series("ekf_gamma_rad", smoothed[:, _GAMMA]),
        pl.Series("ekf_alt_m", smoothed[:, _H]),
        pl.Series("ekf_dtas_ms2", smoothed[:, _DTAS]),
        pl.Series("ekf_dgamma_rads", smoothed[:, _DGAMMA]),
    )


def smooth_flights(
    df: pl.DataFrame,
    *,
    reject_sigma: float = 3.0,
    rolling_window: int = 17,
) -> pl.DataFrame:
    """Apply EKF smoothing to all flights in a Delta Table.

    Partitions by ``meta_flight_id``, applies :func:`smooth_flight` to
    each partition, then concatenates.  Existing ``ekf_*`` output columns
    are dropped before recomputation.

    Args:
        df: Full Delta Table DataFrame.
        reject_sigma: Per-component outlier gating threshold.
        rolling_window: Window size for empirical R estimation.

    Returns:
        DataFrame with ``ekf_*`` columns added to every flight.
    """
    import polars as pl

    # Drop existing output columns for idempotency
    ekf_out = [c for c in df.columns if c.startswith("ekf_") and not c.startswith("ekf_input_")]
    if ekf_out:
        df = df.drop(ekf_out)

    flights = df.partition_by("meta_flight_id", maintain_order=True)
    processed: list[pl.DataFrame] = []
    for flight_df in flights:
        result = smooth_flight(
            flight_df,
            reject_sigma=reject_sigma,
            rolling_window=rolling_window,
        )
        processed.append(result)

    return pl.concat(processed, how="diagonal_relaxed")
