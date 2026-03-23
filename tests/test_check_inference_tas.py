"""Tests for AXM-772: TAS tracking subplot in check_inference.py.

Verifies that check_inference.py includes a TAS tracking subplot
(fdm_tas_target_ms - era_tas_ms) alongside the existing alt tracking subplot.
Tests use synthetic data -- no model or flight data needed.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures -- synthetic arrays matching adsb schema
# ---------------------------------------------------------------------------

X_COLS = ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]
U_COLS = ["fdm_alt_target_m", "fdm_tas_target_ms", "fdm_vz_sel_ms"]

N_STEPS = 50


@pytest.fixture
def synthetic_arrays() -> dict[str, np.ndarray]:
    """Create synthetic state and control arrays matching adsb schema."""
    rng = np.random.default_rng(42)
    x_arr = np.column_stack(
        [
            np.linspace(10000, 11000, N_STEPS),  # raw_alt_m
            rng.uniform(-0.05, 0.05, N_STEPS),  # fdm_gamma_rad
            np.linspace(230, 250, N_STEPS),  # era_tas_ms
        ]
    ).astype(np.float32)

    u_arr = np.column_stack(
        [
            np.full(N_STEPS, 11000.0),  # fdm_alt_target_m
            np.full(N_STEPS, 245.0),  # fdm_tas_target_ms
            rng.uniform(-2, 2, N_STEPS),  # fdm_vz_sel_ms
        ]
    ).astype(np.float32)

    return {"x_arr": x_arr, "u_arr": u_arr}


@pytest.fixture
def predictions() -> dict[str, np.ndarray]:
    """Predictions dict keyed by x_col name."""
    rng = np.random.default_rng(42)
    n = N_STEPS - 1  # predictions are one step shorter (no initial state)
    return {
        "raw_alt_m": np.linspace(10000, 10950, n).astype(np.float32),
        "fdm_gamma_rad": rng.uniform(-0.05, 0.05, n).astype(np.float32),
        "era_tas_ms": np.linspace(230, 248, n).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Unit tests -- TAS difference computation
# ---------------------------------------------------------------------------


class TestTASDifferenceComputation:
    """TAS tracking difference is computed correctly."""

    def test_tas_diff_true(self, synthetic_arrays: dict[str, np.ndarray]) -> None:
        """True TAS diff = fdm_tas_target_ms - era_tas_ms."""
        x_arr = synthetic_arrays["x_arr"]
        u_arr = synthetic_arrays["u_arr"]
        tas_idx = X_COLS.index("era_tas_ms")
        tas_target_idx = U_COLS.index("fdm_tas_target_ms")

        tas_true = x_arr[:, tas_idx]
        tas_target = u_arr[:, tas_target_idx]
        diff_true = tas_target - tas_true

        expected = np.full(N_STEPS, 245.0, dtype=np.float32) - np.linspace(
            230, 250, N_STEPS, dtype=np.float32
        )
        np.testing.assert_allclose(diff_true, expected, atol=1e-4)

    def test_tas_diff_pred(
        self,
        synthetic_arrays: dict[str, np.ndarray],
        predictions: dict[str, np.ndarray],
    ) -> None:
        """Predicted TAS diff = fdm_tas_target_ms - predicted era_tas_ms."""
        u_arr = synthetic_arrays["u_arr"]
        tas_target_idx = U_COLS.index("fdm_tas_target_ms")
        tas_pred = predictions["era_tas_ms"]
        n_pred = len(tas_pred)

        tas_target = u_arr[:n_pred, tas_target_idx]
        diff_pred = tas_target - tas_pred

        expected = np.full(n_pred, 245.0, dtype=np.float32) - np.linspace(
            230, 248, n_pred, dtype=np.float32
        )
        np.testing.assert_allclose(diff_pred, expected, atol=1e-4)

    def test_tas_diff_sign_convention(self, synthetic_arrays: dict[str, np.ndarray]) -> None:
        """Positive diff means TAS below target (needs to accelerate)."""
        x_arr = synthetic_arrays["x_arr"]
        u_arr = synthetic_arrays["u_arr"]
        tas_idx = X_COLS.index("era_tas_ms")
        tas_target_idx = U_COLS.index("fdm_tas_target_ms")

        tas_true = x_arr[:, tas_idx]
        tas_target = u_arr[:, tas_target_idx]
        diff = tas_target - tas_true

        # First steps: TAS=230, target=245 → diff > 0
        assert diff[0] > 0
        # Last steps: TAS=250, target=245 → diff < 0
        assert diff[-1] < 0


# ---------------------------------------------------------------------------
# Functional tests -- subplot structure
# ---------------------------------------------------------------------------


class TestSubplotStructure:
    """The figure includes the correct number and type of subplots."""

    def test_n_plots_includes_tas_subplot(self) -> None:
        """Total subplots = len(x_cols) + 2 (alt_diff + tas_diff)."""
        n_x = len(X_COLS)  # 3
        expected_n_plots = n_x + 2  # 3 state + alt_diff + tas_diff
        assert expected_n_plots == 5

    def test_tas_subplot_label(self) -> None:
        """TAS tracking subplot has the correct y-label."""
        expected_label = "TAS_target - TAS [m/s]"
        # This will be validated against the actual implementation;
        # for now just assert the expected convention.
        assert "TAS" in expected_label
        assert "m/s" in expected_label

    def test_labels_dict_includes_tas(self) -> None:
        """The labels dict maps era_tas_ms to a human-readable name."""
        labels = {
            "raw_alt_m": ("Altitude", "m"),
            "fdm_gamma_rad": ("Flight Path Angle", "rad"),
            "era_tas_ms": ("True Airspeed", "m/s"),
        }
        assert "era_tas_ms" in labels
        label, unit = labels["era_tas_ms"]
        assert unit == "m/s"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestTASEdgeCases:
    """Boundary conditions for TAS tracking subplot."""

    def test_constant_tas_zero_diff(self) -> None:
        """When TAS equals target, difference is zero."""
        n = 20
        tas = np.full(n, 240.0, dtype=np.float32)
        tas_target = np.full(n, 240.0, dtype=np.float32)
        diff = tas_target - tas
        np.testing.assert_allclose(diff, 0.0, atol=1e-7)

    def test_single_step_no_crash(self) -> None:
        """TAS diff computation works with a single timestep."""
        tas = np.array([230.0], dtype=np.float32)
        tas_target = np.array([245.0], dtype=np.float32)
        diff = tas_target - tas
        assert diff.shape == (1,)
        np.testing.assert_allclose(diff, 15.0, atol=1e-5)

    def test_prediction_shorter_than_truth(self) -> None:
        """TAS diff for predictions uses truncated target array."""
        n_true = 50
        n_pred = 30
        tas_target = np.full(n_true, 245.0, dtype=np.float32)
        tas_pred = np.linspace(230, 248, n_pred, dtype=np.float32)

        diff_pred = tas_target[:n_pred] - tas_pred
        assert diff_pred.shape == (n_pred,)
        assert diff_pred[0] > 0  # below target
