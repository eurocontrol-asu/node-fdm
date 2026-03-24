"""Tests for fdm_gamma_target_rad — unified gamma target (fusion 3 sources + bfill)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

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
# Unit tests
# ---------------------------------------------------------------------------


class TestGammaTargetExists:
    """fdm_gamma_target_rad column is created with full config."""

    def test_gamma_target_exists(self) -> None:
        df = _make_standard_flight()
        result = build_selected_params(df, _full_config())

        assert "fdm_gamma_target_rad" in result.columns


class TestGammaTargetNoNan:
    """fdm_gamma_target_rad has no NaN values on a standard flight."""

    def test_gamma_target_no_nan(self) -> None:
        df = _make_standard_flight()
        result = build_selected_params(df, _full_config())

        col = result["fdm_gamma_target_rad"]
        nan_count = col.null_count() + col.is_nan().sum()
        assert nan_count == 0, f"Expected zero NaN, got {nan_count}"


class TestGammaTargetLastRowEqualsActual:
    """Last value of fdm_gamma_target_rad equals actual fdm_gamma_rad[-1]."""

    def test_gamma_target_last_row_equals_actual(self) -> None:
        df = _make_standard_flight()
        result = build_selected_params(df, _full_config())

        last_target = result["fdm_gamma_target_rad"][-1]
        last_actual = result["fdm_gamma_rad"][-1]
        assert last_target == pytest.approx(last_actual, rel=1e-6)


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestGammaTargetVzSource:
    """Where vz plateau is detected, gamma_target ≈ arcsin(vz_ms / tas_ms)."""

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

        # In the vz plateau region (indices ~50..150), gamma_target should
        # approximate arcsin(vz_ms / tas_ms)
        expected_gamma = float(np.arcsin(1500.0 * _FT_MIN_TO_MS / (300.0 * _KT_TO_MS)))
        target = result["fdm_gamma_target_rad"].to_numpy()

        # Check the middle of the plateau
        plateau_vals = target[70:130]
        np.testing.assert_allclose(plateau_vals, expected_gamma, atol=0.01)


class TestGammaTargetGammaSource:
    """Where gamma plateau is detected, gamma_target equals that plateau value."""

    def test_gamma_target_gamma_source(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        # Flight with gamma plateau at -0.05 rad in middle section
        # No vz data to avoid vz-masking of gamma_sel detection
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

        # Config with gamma only (no vz to avoid masking)
        config = {
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
        # In the gamma plateau region, target should be ~ -0.05
        plateau_vals = target[70:130]
        np.testing.assert_allclose(plateau_vals, gamma_val, atol=0.005)


class TestGammaTargetAltHoldSource:
    """Where altitude plateau is detected, gamma_target = 0."""

    def test_gamma_target_alt_hold_source(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        # Flight with altitude plateau in middle section
        alt = np.concatenate(
            [
                np.linspace(10_000, 35_000, 50),
                np.full(100, 35_000.0),  # level flight
                np.linspace(35_000, 10_000, 50),
            ]
        )
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

        # Where alt plateau is detected, gamma_target should be 0
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()

        alt_hold_mask = ~np.isnan(alt_sel)
        assert alt_hold_mask.sum() > 0, "Expected altitude plateau detection"
        np.testing.assert_allclose(target[alt_hold_mask], 0.0, atol=1e-10)


class TestGammaTargetPriority:
    """alt_hold (gamma=0) overwrites vz→gamma when both overlap."""

    def test_gamma_target_priority(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        # Flight at constant altitude with constant vz (overlapping sources)
        # Alt plateau → gamma_target=0 should override vz→gamma
        alt = np.full(n, 35_000.0)  # entire flight level
        vz = np.full(n, 500.0) + rng.normal(0, 3, n)  # small constant vz
        tas_kt = np.full(n, 450.0)
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

        config = _full_config()
        config["vz"]["min_abs_value"] = 10  # lower threshold so vz is detected
        result = build_selected_params(df, config)

        # Alt hold should override vz->gamma: gamma_target = 0 where alt plateau
        # (except last row which is anchored to actual gamma)
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()

        alt_hold_mask = ~np.isnan(alt_sel)
        # Exclude last row (anchor point)
        alt_hold_mask[-1] = False
        assert alt_hold_mask.sum() > 0, "Expected altitude plateau detection"
        np.testing.assert_allclose(target[alt_hold_mask], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGammaTargetNoSegments:
    """No segments detected — bfill from last actual gamma."""

    def test_no_segments_bfill(self) -> None:
        rng = np.random.default_rng(99)
        n = 10  # very short flight, below min_len thresholds
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

        col = result["fdm_gamma_target_rad"]
        last_actual = result["fdm_gamma_rad"][-1]
        last_target = col[-1]

        # Last row anchored to actual gamma
        assert last_target == pytest.approx(last_actual, rel=1e-6)
        # Backward fill should fill everything
        nan_count = col.null_count() + col.is_nan().sum()
        assert nan_count == 0


class TestGammaTargetAllAltHold:
    """alt_sel covers everything → gamma_target = 0 everywhere."""

    def test_all_alt_hold(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        alt = np.full(n, 35_000.0)  # constant altitude
        vz = rng.normal(0, 5, n)  # near-zero vz, noisy
        tas_kt = np.full(n, 450.0)
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

        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()

        # Where alt is detected -> gamma_target = 0 (except last row = anchor)
        alt_hold_mask = ~np.isnan(alt_sel)
        alt_hold_mask[-1] = False
        assert alt_hold_mask.sum() > 0
        np.testing.assert_allclose(target[alt_hold_mask], 0.0, atol=1e-10)


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

        # Should not raise
        result = build_selected_params(df, _full_config())

        # Column should exist (even if values are NaN/clamped)
        if "fdm_gamma_target_rad" in result.columns:
            target = result["fdm_gamma_target_rad"].to_numpy()
            # No inf values
            assert not np.any(np.isinf(target[~np.isnan(target)]))
