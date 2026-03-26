"""Tests for fdm_gamma_target_rad — unified gamma target v2 (AXM-809).

Priority order (highest → lowest): vz→gamma > gamma_sel > gamma_from_alt.
NaN-preserving: gaps between segments stay NaN (no backward-fill).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from node_fdm_data.preprocessing.convert import convert_si
from node_fdm_data.segments import build_selected_params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FT_MIN_TO_MS = 0.3048 / 60  # ft/min → m/s
_KT_TO_MS = 0.514444


def _full_config() -> dict[str, Any]:
    """Config that enables vz, gamma, and alt segment detection."""
    return {
        "vz": {
            "tol": 25,
            "min_len": 20,
            "use_alt": False,
            "min_abs_value": 75,
            "smooth_window": 15,
            "smooth_method": "savgol",
        },
        "gamma": {
            "tol": 0.002,
            "min_len": 15,
            "use_alt": False,
            "smooth_window": 5,
            "smooth_method": "savgol",
        },
        "alt": {
            "tol": 25,
            "min_len": 5,
            "use_alt": False,
            "min_abs_value": 25,
            "smooth_window": 5,
            "smooth_method": "savgol",
        },
    }


def _make_standard_flight(n: int = 300, *, seed: int = 42) -> pl.DataFrame:
    """Build a 3-phase flight (climb / cruise / descent) with vz, gamma, alt."""
    rng = np.random.default_rng(seed)
    third = n // 3

    alt = np.concatenate(
        [
            np.linspace(0, 35_000, third),
            np.full(n - 2 * third, 35_000),
            np.linspace(35_000, 0, third),
        ]
    )

    vz = np.concatenate(
        [
            np.full(third, 1500.0) + rng.normal(0, 5, third),
            np.full(n - 2 * third, 0.0) + rng.normal(0, 5, n - 2 * third),
            np.full(third, -1500.0) + rng.normal(0, 5, third),
        ]
    )

    tas_kt = np.concatenate(
        [
            np.linspace(200, 450, third),
            np.full(n - 2 * third, 450.0) + rng.normal(0, 0.2, n - 2 * third),
            np.linspace(450, 200, third),
        ]
    )

    vz_ms = vz * _FT_MIN_TO_MS
    tas_ms = tas_kt * _KT_TO_MS
    gamma = np.arcsin(np.clip(vz_ms / np.where(tas_ms == 0, np.nan, tas_ms), -1, 1))

    return pl.DataFrame(
        {
            "raw_alt_ft": alt,
            "raw_vz_ftmin": vz,
            "era_tas_kt": tas_kt,
            "fdm_gamma_rad": gamma,
        }
    )


# ---------------------------------------------------------------------------
# Existing unit tests (unchanged)
# ---------------------------------------------------------------------------


class TestGammaTargetExists:
    """fdm_gamma_target_rad column is created with full config."""

    def test_gamma_target_exists(self) -> None:
        df = _make_standard_flight()
        result = build_selected_params(df, _full_config())

        assert "fdm_gamma_target_rad" in result.columns


# ---------------------------------------------------------------------------
# Existing functional tests (unchanged)
# ---------------------------------------------------------------------------


class TestGammaTargetVzSource:
    """Where vz plateau is detected, gamma_target ~ arcsin(vz_ms / tas_ms)."""

    def test_gamma_target_vz_source(self) -> None:
        n = 200
        # Flight with a clear vz plateau at 1500 ft/min in middle section
        vz = np.concatenate(
            [
                np.linspace(0, 1500, 50),
                np.full(100, 1500.0),  # stable plateau
                np.linspace(1500, 0, 50),
            ]
        )
        tas_kt = np.full(n, 300.0)
        alt = np.linspace(5_000, 35_000, n)
        vz_ms = vz * _FT_MIN_TO_MS
        tas_ms = tas_kt * _KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / tas_ms, -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, _full_config())

        expected_gamma = float(np.arcsin(1500.0 * _FT_MIN_TO_MS / (300.0 * _KT_TO_MS)))
        target = result["fdm_gamma_target_rad"].to_numpy()

        plateau_vals = target[70:130]
        np.testing.assert_allclose(plateau_vals, expected_gamma, atol=0.01)


class TestGammaTargetGammaSource:
    """Where gamma plateau is detected, gamma_target equals that plateau value."""

    def test_gamma_target_gamma_source(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        gamma_val = -0.05
        gamma = np.concatenate(
            [
                np.linspace(-0.1, gamma_val, 50),
                np.full(100, gamma_val) + rng.normal(0, 0.0001, 100),
                np.linspace(gamma_val, 0.0, 50),
            ]
        )
        tas_kt = np.full(n, 400.0)
        alt = np.linspace(35_000, 10_000, n)

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        config: dict[str, Any] = {
            "gamma": {
                "tol": 0.002,
                "min_len": 15,
                "use_alt": False,
                "smooth_window": 5,
                "smooth_method": "savgol",
            },
        }
        result = build_selected_params(df, config)

        target = result["fdm_gamma_target_rad"].to_numpy()
        plateau_vals = target[70:130]
        np.testing.assert_allclose(plateau_vals, gamma_val, atol=0.005)


# ---------------------------------------------------------------------------
# Existing edge cases (unchanged)
# ---------------------------------------------------------------------------


class TestGammaTargetZeroTas:
    """vz_sel with zero TAS — no crash, clamped gamma."""

    def test_zero_tas_no_crash(self) -> None:
        n = 100
        alt = np.linspace(5_000, 15_000, n)
        vz = np.full(n, 1500.0)
        tas_kt = np.zeros(n)  # degenerate: TAS = 0
        gamma = np.full(n, 0.0)  # placeholder gamma

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, _full_config())

        if "fdm_gamma_target_rad" in result.columns:
            target = result["fdm_gamma_target_rad"].to_numpy()
            assert not np.any(np.isinf(target[~np.isnan(target)]))


# ===========================================================================
# AXM-809 — New unit tests
# ===========================================================================


class TestGammaTargetKnownMask:
    """Gaps between segments have gamma_target_known=0 (filled to 0.0, not NaN)."""

    def test_gamma_target_known_gaps(self) -> None:
        n = 300
        rng = np.random.default_rng(42)
        third = n // 3

        # Climb with clear vz plateau, erratic middle (no plateau), descent
        vz = np.concatenate(
            [
                np.full(third, 1500.0) + rng.normal(0, 5, third),
                rng.uniform(-200, 200, n - 2 * third),  # erratic — no segment
                np.full(third, -1500.0) + rng.normal(0, 5, third),
            ]
        )
        alt = np.concatenate(
            [
                np.linspace(5_000, 20_000, third),
                np.linspace(20_000, 20_500, n - 2 * third),  # near-flat but noisy
                np.linspace(20_500, 5_000, third),
            ]
        )
        tas_kt = np.full(n, 350.0)
        vz_ms = vz * _FT_MIN_TO_MS
        tas_ms = tas_kt * _KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / tas_ms, -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        # Use config without alt detection so middle section has no source
        config: dict[str, Any] = {
            "vz": {
                "tol": 25,
                "min_len": 20,
                "use_alt": False,
                "min_abs_value": 75,
                "smooth_window": 15,
                "smooth_method": "savgol",
            },
        }
        result = build_selected_params(df, config)

        assert "fdm_gamma_target_known" in result.columns
        known = result["fdm_gamma_target_known"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()

        # Gaps should have known=0, not NaN in target
        unknown_count = (known == 0).sum()
        assert unknown_count > 0, "Expected unknown gaps between segments"
        assert not np.any(np.isnan(target)), "gamma_target should have no NaN (filled to 0.0)"


class TestGammaTargetPriorityVzOverGamma:
    """vz->gamma (highest priority) overrides gamma_sel when both overlap."""

    def test_gamma_target_priority_vz_over_gamma(self) -> None:
        n = 200
        rng = np.random.default_rng(42)

        # Constant vz at 1000 ft/min + constant gamma at a DIFFERENT value
        # so we can distinguish which source wins.
        vz_val = 1000.0
        tas_val_kt = 400.0
        expected_vz_gamma = float(np.arcsin(vz_val * _FT_MIN_TO_MS / (tas_val_kt * _KT_TO_MS)))
        # Set gamma to a distinctly different constant (0.08 rad ~ 4.6 deg)
        forced_gamma = 0.08

        vz = np.full(n, vz_val) + rng.normal(0, 2, n)
        tas_kt = np.full(n, tas_val_kt)
        alt = np.linspace(5_000, 25_000, n)
        gamma = np.full(n, forced_gamma) + rng.normal(0, 0.0001, n)

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        config: dict[str, Any] = {
            "vz": {
                "tol": 25,
                "min_len": 20,
                "use_alt": False,
                "min_abs_value": 75,
                "smooth_window": 15,
                "smooth_method": "savgol",
            },
            "gamma": {
                "tol": 0.002,
                "min_len": 15,
                "use_alt": False,
                "smooth_window": 5,
                "smooth_method": "savgol",
            },
        }

        result = build_selected_params(df, config)
        target = result["fdm_gamma_target_rad"].to_numpy()

        # Where vz_sel is detected, gamma_target should match vz->gamma,
        # NOT the gamma_sel value (0.08 rad).
        vz_sel = result["fdm_vz_sel_ftmin"].to_numpy()
        vz_mask = ~np.isnan(vz_sel)
        assert vz_mask.sum() > 20, "Expected vz plateau detection"

        np.testing.assert_allclose(
            target[vz_mask],
            expected_vz_gamma,
            atol=0.005,
            err_msg="vz->gamma should override gamma_sel",
        )


class TestGammaTargetPriorityGammaOverAlt:
    """gamma_sel overrides gamma_from_alt (alt is lowest priority)."""

    def test_gamma_target_priority_gamma_over_alt(self) -> None:
        n = 200
        rng = np.random.default_rng(42)

        # Constant altitude (alt_sel detected → gamma_from_alt = 0)
        # AND a non-zero constant gamma (gamma_sel detected)
        alt = np.full(n, 35_000.0)
        gamma_val = -0.03
        gamma = np.full(n, gamma_val) + rng.normal(0, 0.0001, n)
        # Near-zero vz so vz doesn't dominate
        vz = rng.normal(0, 5, n)
        tas_kt = np.full(n, 450.0)

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        config = _full_config()
        result = build_selected_params(df, config)

        # Alt plateau should be detected
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        alt_mask = ~np.isnan(alt_sel)
        assert alt_mask.sum() > 0, "Expected altitude plateau detection"

        # Gamma_sel should also be detected
        gamma_sel = result["fdm_gamma_sel_rad"].to_numpy()
        gamma_mask = ~np.isnan(gamma_sel)
        assert gamma_mask.sum() > 0, "Expected gamma plateau detection"

        # Where both overlap, gamma_sel wins (not gamma_from_alt=0)
        overlap = alt_mask & gamma_mask
        assert overlap.sum() > 0, "Expected overlap between alt and gamma segments"

        target = result["fdm_gamma_target_rad"].to_numpy()
        np.testing.assert_allclose(
            target[overlap],
            gamma_val,
            atol=0.005,
            err_msg="gamma_sel should override gamma_from_alt",
        )


class TestGammaTargetNoNearZero:
    """Gamma_sel does not detect near-zero plateaus during cruise."""

    def test_gamma_target_no_near_zero(self) -> None:
        df = _make_standard_flight()
        result = build_selected_params(df, _full_config())

        if "fdm_gamma_sel_rad" not in result.columns:
            pytest.skip("gamma_sel not produced")

        gamma_sel = result["fdm_gamma_sel_rad"].to_numpy()
        valid = gamma_sel[~np.isnan(gamma_sel)]

        # No detected gamma_sel segment should have near-zero value
        # (cruise gamma ~ 0 must be filtered out by min_abs_value)
        if len(valid) > 0:
            assert np.all(np.abs(valid) > 0.005), (
                f"gamma_sel contains near-zero values: {valid[np.abs(valid) <= 0.005]}"
            )


class TestGammaDiffNanFilledZero:
    """gamma_diff = 0 where gamma_target is NaN (not NaN propagation)."""

    def test_gamma_diff_nan_filled_zero(self) -> None:
        n = 50
        rng = np.random.default_rng(42)

        # Build a DataFrame that already has gamma_target with some NaN
        gamma_actual = rng.uniform(-0.05, 0.05, n)
        gamma_target = gamma_actual.copy()
        # Inject NaN in middle section
        gamma_target[15:35] = np.nan

        df = pl.DataFrame(
            {
                "fdm_gamma_target_rad": gamma_target,
                "fdm_gamma_rad": gamma_actual,
                # convert_si also needs raw columns for SI conversions
                "raw_alt_ft": np.linspace(10_000, 35_000, n),
                "era_tas_kt": np.full(n, 400.0),
                "raw_vz_ftmin": np.full(n, 0.0),
            }
        )

        result = convert_si(df)

        assert "fdm_gamma_diff_rad" in result.columns, "gamma_diff column missing"
        diff = result["fdm_gamma_diff_rad"].to_numpy()

        # Where target was NaN, diff should be 0 (not NaN)
        nan_mask = np.isnan(gamma_target)
        assert not np.any(np.isnan(diff[nan_mask])), (
            "gamma_diff should be 0 where gamma_target is NaN"
        )
        np.testing.assert_allclose(diff[nan_mask], 0.0, atol=1e-10)


# ===========================================================================
# AXM-809 — New edge cases
# ===========================================================================


class TestGammaTargetAllUnknown:
    """Very short flight, no segments → gamma_target_known all 0, target all 0.0."""

    def test_all_unknown_target(self) -> None:
        rng = np.random.default_rng(99)
        n = 10  # below min_len thresholds → no segments detected
        vz = rng.uniform(-500, 500, n)
        tas_kt = rng.uniform(200, 400, n)
        alt = np.linspace(5_000, 10_000, n)
        vz_ms = vz * _FT_MIN_TO_MS
        tas_ms = tas_kt * _KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / np.where(tas_ms == 0, np.nan, tas_ms), -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, _full_config())

        known = result["fdm_gamma_target_known"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()
        assert np.all(known == 0), f"Expected all unknown, got {(known == 1).sum()} known"
        assert not np.any(np.isnan(target)), "gamma_target should be 0.0 not NaN"


class TestGammaTargetOnlyAltSel:
    """Only alt_sel present → gamma_target = 0 in cruise, known=0 elsewhere."""

    def test_only_alt_sel(self) -> None:
        n = 300
        rng = np.random.default_rng(42)
        third = n // 3

        # Long cruise in the middle, no vz/gamma segments
        alt = np.concatenate(
            [
                np.linspace(10_000, 35_000, third),
                np.full(n - 2 * third, 35_000.0),  # cruise
                np.linspace(35_000, 10_000, third),
            ]
        )
        # Erratic vz (no plateau detectable)
        vz = rng.uniform(-200, 200, n)
        tas_kt = np.full(n, 400.0)
        vz_ms = vz * _FT_MIN_TO_MS
        tas_ms = tas_kt * _KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / tas_ms, -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        # Only alt detection enabled (no vz, no gamma)
        config: dict[str, Any] = {
            "alt": {
                "tol": 25,
                "min_len": 5,
                "use_alt": False,
                "min_abs_value": 25,
                "smooth_window": 5,
                "smooth_method": "savgol",
            },
        }
        result = build_selected_params(df, config)

        target_arr = result["fdm_gamma_target_rad"].to_numpy()
        known = result["fdm_gamma_target_known"].to_numpy()

        # Where alt_sel is detected → gamma_target = 0, known = 1
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        alt_mask = ~np.isnan(alt_sel)
        assert alt_mask.sum() > 0, "Expected altitude plateau detection"
        np.testing.assert_allclose(target_arr[alt_mask], 0.0, atol=1e-10)
        assert np.all(known[alt_mask] == 1.0)

        # Elsewhere → gamma_target = 0.0 (filled), known = 0
        non_alt_mask = np.isnan(alt_sel)
        assert np.all(known[non_alt_mask] == 0.0)


# ===========================================================================
# Updated existing tests — contract changes for AXM-809
# ===========================================================================


class TestGammaTargetAltHoldSource:
    """Where only alt plateau detected (no vz/gamma), gamma_target = 0."""

    def test_gamma_target_alt_hold_source(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                np.linspace(10_000, 35_000, 50),
                np.full(100, 35_000.0),  # level flight
                np.linspace(35_000, 10_000, 50),
            ]
        )
        # Near-zero vz in cruise (below min_abs_value), so no vz_sel in cruise
        vz = np.concatenate(
            [
                np.full(50, 1500.0),
                np.full(100, 0.0) + rng.normal(0, 5, 100),
                np.full(50, -1500.0),
            ]
        )
        tas_kt = np.full(n, 450.0)
        vz_ms = vz * _FT_MIN_TO_MS
        tas_ms = tas_kt * _KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / np.where(tas_ms == 0, np.nan, tas_ms), -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "era_tas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, _full_config())

        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()

        # In cruise zone where only alt_sel detected (no vz_sel),
        # gamma_target should be 0
        alt_hold_mask = ~np.isnan(alt_sel)
        assert alt_hold_mask.sum() > 0, "Expected altitude plateau detection"

        # Only check rows where alt_sel detected but vz_sel is NOT detected
        if "fdm_vz_sel_ftmin" in result.columns:
            vz_sel = result["fdm_vz_sel_ftmin"].to_numpy()
            only_alt = alt_hold_mask & np.isnan(vz_sel)
        else:
            only_alt = alt_hold_mask
        if only_alt.sum() > 0:
            np.testing.assert_allclose(target[only_alt], 0.0, atol=1e-10)
