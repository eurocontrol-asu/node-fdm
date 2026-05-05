"""Tests for altitude-plateau gamma_target=0 detection (AXM-802).

Where ``fdm_alt_sel_ft`` is not NaN the aircraft is in level flight
(ALT HLD) and the flight-path-angle target is zero.  The new column
``fdm_gamma_from_alt_rad`` encodes this: 0.0 inside altitude plateaus,
NaN elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from node_fdm_data.segments import build_selected_params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alt_config() -> dict[str, Any]:
    """Minimal config that enables altitude segment detection."""
    return {
        "alt": {"tol": 25, "min_len": 5, "use_alt": False},
    }


def _full_config() -> dict[str, Any]:
    """Config with mach + alt sections."""
    return {
        "mach": {
            "tol": 0.0005,
            "min_len": 30,
            "alt_threshold": 15000,
            "smooth_window": 10,
            "use_alt": True,
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


# ---------------------------------------------------------------------------
# Unit tests — column semantics
# ---------------------------------------------------------------------------


class TestGammaFromAltColumn:
    """fdm_gamma_from_alt_rad column: 0.0 in level-flight, NaN elsewhere."""

    def test_column_created_when_alt_segments_exist(self) -> None:
        """fdm_gamma_from_alt_rad is produced when alt config is present."""
        alt = np.concatenate(
            [
                np.linspace(5000, 35000, 50),  # climb
                np.full(50, 35000.0),  # level
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())

        assert "fdm_gamma_from_alt_rad" in result.columns

    def test_zero_inside_altitude_plateau(self) -> None:
        """After relaxation zone, level-flight rows have gamma_from_alt = 0.0."""
        n = 100
        alt = np.full(n, 35000.0)  # entire flight at constant altitude
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())

        alt_sel = result["fdm_alt_sel_ft"]
        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()

        # Where alt_sel is detected (not NaN), gamma is 0.0 after relaxation zone
        level_mask = ~alt_sel.is_nan()
        assert level_mask.sum() > 0, "expected altitude plateau detection"
        # Skip first 15 relaxation timesteps — those are NaN
        stabilised = gamma[15:]
        assert np.nansum(stabilised == 0.0) > 0, "gamma must be 0.0 after relaxation zone"

    def test_nan_outside_altitude_plateau(self) -> None:
        """Rows where fdm_alt_sel_ft is NaN have gamma_from_alt = NaN."""
        alt = np.concatenate(
            [
                np.linspace(5000, 35000, 50),  # climb (no plateau)
                np.full(50, 35000.0),  # level plateau
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())

        alt_sel = result["fdm_alt_sel_ft"]
        gamma = result["fdm_gamma_from_alt_rad"]

        non_level_mask = alt_sel.is_nan()
        assert non_level_mask.sum() > 0, "expected some non-level rows"
        gamma_outside = gamma.filter(non_level_mask)
        assert gamma_outside.is_nan().all(), "gamma must be NaN outside level-flight regions"

    def test_exact_alignment_with_alt_sel(self) -> None:
        """fdm_gamma_from_alt_rad is 0.0 where fdm_alt_sel_ft is not NaN, after relaxation."""
        alt = np.concatenate(
            [
                np.full(60, 10000.0),  # level at 10k
                np.linspace(10000, 35000, 80),  # climb
                np.full(60, 35000.0),  # level at 35k
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())

        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()

        level_mask = ~np.isnan(alt_sel)
        # Non-level rows → NaN
        assert np.all(np.isnan(gamma[~level_mask]))
        # Level rows: first 15 per segment are NaN (relaxation), rest are 0.0
        level_vals = gamma[level_mask]
        n_nan = np.isnan(level_vals).sum()
        n_zero = (level_vals == 0.0).sum()
        assert n_nan > 0, "expected relaxation NaN at start of segments"
        assert n_zero > 0, "expected 0.0 after relaxation zone"
        assert n_nan + n_zero == len(level_vals), "level rows must be NaN or 0.0"


# ---------------------------------------------------------------------------
# Functional tests — integration with full pipeline
# ---------------------------------------------------------------------------


class TestGammaFromAltIntegration:
    """Integration: gamma_from_alt inside full build_selected_params."""

    def test_full_config_produces_column(self) -> None:
        """Full config with mach + alt produces fdm_gamma_from_alt_rad."""
        n = 300
        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                np.linspace(0, 35000, 100),
                np.full(100, 35000),
                np.linspace(35000, 0, 100),
            ]
        )
        mach = np.concatenate(
            [
                np.linspace(0.3, 0.78, 100),
                np.full(100, 0.78),
                np.linspace(0.78, 0.3, 100),
            ]
        )
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "era_mach": mach + rng.normal(0, 0.0001, n),
            }
        )
        result = build_selected_params(df, _full_config())

        assert "fdm_gamma_from_alt_rad" in result.columns
        # Cruise plateau (100 rows) should produce zeros after 15-step relaxation
        gamma = result["fdm_gamma_from_alt_rad"]
        assert (gamma == 0.0).sum() > 0, (
            "cruise level-flight should produce gamma=0 after relaxation"
        )

    def test_coexists_with_gamma_sel(self) -> None:
        """fdm_gamma_from_alt_rad and fdm_gamma_sel_rad both exist."""
        n = 300
        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                np.linspace(0, 35000, 100),
                np.full(100, 35000),
                np.linspace(35000, 0, 100),
            ]
        )
        vz = np.concatenate(
            [
                np.full(100, 2000.0),
                np.full(100, 0.0) + rng.normal(0, 5, 100),
                np.full(100, -1500.0),
            ]
        )
        gamma = np.arcsin(np.clip(vz * (0.3048 / 60) / (450 * 0.514444 + 1e-6), -1, 1))
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "era_mach": np.linspace(0.3, 0.78, n),
                "raw_vz_ftmin": vz,
                "fdm_gamma_rad": gamma,
            }
        )
        config = {
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
        result = build_selected_params(df, config)

        assert "fdm_gamma_sel_rad" in result.columns
        assert "fdm_gamma_from_alt_rad" in result.columns

    def test_length_preserved(self) -> None:
        """Output DataFrame has the same length as input."""
        n = 150
        alt = np.concatenate(
            [
                np.full(50, 10000.0),
                np.linspace(10000, 35000, 50),
                np.full(50, 35000.0),
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())
        assert len(result) == n
        assert len(result["fdm_gamma_from_alt_rad"]) == n


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGammaFromAltEdgeCases:
    """Edge cases for altitude-plateau gamma detection."""

    def test_no_alt_config_no_column(self) -> None:
        """Without 'alt' in config, fdm_gamma_from_alt_rad is not created."""
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "era_mach": np.full(n, 0.78),
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
        }
        result = build_selected_params(df, config)
        assert "fdm_gamma_from_alt_rad" not in result.columns

    def test_no_altitude_plateau_all_nan(self) -> None:
        """Monotonic altitude → no alt segments → gamma_from_alt all NaN."""
        n = 100
        alt = np.linspace(5000, 35000, n)  # pure climb, no plateau
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())

        assert "fdm_gamma_from_alt_rad" in result.columns
        assert result["fdm_gamma_from_alt_rad"].is_nan().all()

    def test_entire_flight_level(self) -> None:
        """Constant altitude → gamma_from_alt = 0.0 after relaxation zone."""
        n = 100
        df = pl.DataFrame({"raw_alt_ft": np.full(n, 35000.0)})
        result = build_selected_params(df, _alt_config())

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        level_count = (~np.isnan(alt_sel)).sum()
        zero_count = (gamma == 0.0).sum()
        nan_count = np.isnan(gamma[~np.isnan(alt_sel)]).sum()
        # Relaxation zone (15 timesteps) + stabilised zone = total level rows
        assert zero_count + nan_count == level_count
        assert nan_count <= 15, f"at most 15 relaxation NaN, got {nan_count}"

    def test_multiple_level_segments(self) -> None:
        """Multiple level segments: relaxation NaN at start, then gamma=0."""
        alt = np.concatenate(
            [
                np.full(40, 10000.0),  # level at 10k
                np.linspace(10000, 35000, 40),  # climb
                np.full(40, 35000.0),  # level at 35k
                np.linspace(35000, 20000, 40),  # descent
                np.full(40, 20000.0),  # level at 20k
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()

        level_mask = ~np.isnan(alt_sel)
        assert level_mask.sum() > 0
        level_vals = gamma[level_mask]
        # Each segment of 40 rows: 15 NaN + 25 zeros
        n_zero = (level_vals == 0.0).sum()
        n_nan = np.isnan(level_vals).sum()
        assert n_zero > 0, "expected gamma=0 after relaxation"
        assert n_nan > 0, "expected relaxation NaN at segment starts"
        assert n_nan + n_zero == len(level_vals)

    def test_short_flight_below_min_len(self) -> None:
        """Flight shorter than min_len → no alt segments → all NaN gamma."""
        n = 3
        df = pl.DataFrame({"raw_alt_ft": np.full(n, 35000.0)})
        result = build_selected_params(df, _alt_config())

        assert "fdm_gamma_from_alt_rad" in result.columns
        # min_len=5 > n=3, so no segments detected
        assert result["fdm_gamma_from_alt_rad"].is_nan().all()

    def test_dtype_is_float64(self) -> None:
        """fdm_gamma_from_alt_rad column is Float64."""
        n = 50
        df = pl.DataFrame({"raw_alt_ft": np.full(n, 35000.0)})
        result = build_selected_params(df, _alt_config())

        assert result["fdm_gamma_from_alt_rad"].dtype == pl.Float64

    def test_segment_shorter_than_relax(self) -> None:
        """Segment shorter than relaxation window → entire segment is NaN."""
        alt = np.concatenate(
            [
                np.linspace(5000, 35000, 50),  # climb
                np.full(10, 35000.0),  # level — only 10 rows (< 15 relax)
                np.linspace(35000, 5000, 50),  # descent
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _alt_config())

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        level_mask = ~np.isnan(alt_sel)
        if level_mask.sum() > 0:
            # Segment ≤ 15 → all level rows should be NaN (fully relaxed)
            assert np.all(np.isnan(gamma[level_mask]))

    def test_relax_zero_disables_relaxation(self) -> None:
        """alt_hold_relax=0 → old behaviour, no NaN in level rows."""
        n = 100
        alt = np.full(n, 35000.0)
        df = pl.DataFrame({"raw_alt_ft": alt})
        config = {**_alt_config(), "alt_hold_relax": 0}
        result = build_selected_params(df, config)

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        level_mask = ~np.isnan(alt_sel)
        assert (gamma[level_mask] == 0.0).all(), "relax=0 should give all zeros"
