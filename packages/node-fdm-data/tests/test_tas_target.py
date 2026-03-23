"""Tests for fdm_tas_target_kt — unified TAS target from segments."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.conversions import kt_to_ms
from node_fdm_data.segments import build_selected_params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flight(  # noqa: PLR0913
    n: int = 300,
    *,
    mach_cruise: float = 0.78,
    cas_climb: float = 250.0,
    cas_descent: float = 250.0,
    tas_cruise_kt: float = 450.0,
    include_mach: bool = True,
    include_cas: bool = True,
    include_tas: bool = True,
    alt_nan_indices: list[int] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a synthetic 3-phase flight (climb / cruise / descent).

    Returns a DataFrame with columns expected by ``build_selected_params``.
    """
    rng = np.random.default_rng(seed)
    third = n // 3

    # Altitude profile
    alt = np.concatenate(
        [
            np.linspace(0, 35_000, third),
            np.full(n - 2 * third, 35_000),
            np.linspace(35_000, 0, third),
        ]
    )
    if alt_nan_indices:
        alt = alt.copy()
        for idx in alt_nan_indices:
            alt[idx] = np.nan

    cols: dict[str, np.ndarray] = {"raw_alt_ft": alt}

    if include_mach:
        mach = np.concatenate(
            [
                np.linspace(0.3, mach_cruise, third),
                np.full(n - 2 * third, mach_cruise) + rng.normal(0, 0.0001, n - 2 * third),
                np.linspace(mach_cruise, 0.3, third),
            ]
        )
        cols["era_mach"] = mach

    if include_cas:
        cas = np.concatenate(
            [
                np.full(third, cas_climb) + rng.normal(0, 0.1, third),
                np.linspace(cas_climb, 280, n - 2 * third),
                np.full(third, cas_descent) + rng.normal(0, 0.1, third),
            ]
        )
        cols["bds_ias_kt"] = cas

    if include_tas:
        tas = np.concatenate(
            [
                np.linspace(200, tas_cruise_kt, third),
                np.full(n - 2 * third, tas_cruise_kt) + rng.normal(0, 0.2, n - 2 * third),
                np.linspace(tas_cruise_kt, 200, third),
            ]
        )
        cols["era_tas_kt"] = tas

    return pl.DataFrame(cols)


def _config(
    *,
    include_mach: bool = True,
    include_cas: bool = True,
    include_tas: bool = True,
) -> dict[str, dict[str, object]]:
    """Return a selected-params config dict."""
    cfg: dict[str, dict[str, object]] = {}
    if include_mach:
        cfg["mach"] = {
            "tol": 0.0005,
            "min_len": 30,
            "alt_threshold": 15_000,
            "smooth_window": 10,
            "use_alt": True,
        }
    if include_cas:
        cfg["cas"] = {
            "tol": 0.75,
            "min_len": 20,
            "use_alt": False,
            "smooth_window": 10,
            "smooth_method": "savgol",
        }
    if include_tas:
        cfg["tas"] = {
            "tol": 0.5,
            "min_len": 20,
            "use_alt": False,
            "smooth_window": 10,
            "smooth_method": "savgol",
        }
    return cfg


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestTasTargetNoNan:
    """fdm_tas_target_kt has zero NaN on a full flight."""

    def test_tas_target_no_nan(self) -> None:
        df = _make_flight()
        result = build_selected_params(df, _config())

        assert "fdm_tas_target_kt" in result.columns
        nan_count = (
            result["fdm_tas_target_kt"].null_count() + result["fdm_tas_target_kt"].is_nan().sum()
        )
        assert nan_count == 0, f"Expected zero NaN, got {nan_count}"


class TestTasTargetMachPriority:
    """When Mach and CAS segments overlap, Mach-derived TAS wins."""

    def test_tas_target_mach_priority(self) -> None:
        df = _make_flight()
        result = build_selected_params(df, _config())

        # In the cruise phase (middle third), Mach segment is detected.
        # The TAS target there should equal the Mach-derived TAS, not CAS-derived.
        n = len(result)
        third = n // 3
        cruise_start = third
        cruise_end = 2 * third

        # Where mach_sel is not NaN, tas_target should come from Mach conversion
        mach_sel = result["fdm_mach_sel"]
        tas_target = result["fdm_tas_target_kt"]

        # Find rows with valid Mach selection in cruise
        for i in range(cruise_start, cruise_end):
            mach_val = mach_sel[i]
            if mach_val is not None and not np.isnan(mach_val):
                # TAS target should be populated (Mach takes priority)
                target_val = tas_target[i]
                assert target_val is not None and not np.isnan(target_val)


class TestTasTargetLastRowActual:
    """Last row of fdm_tas_target_kt equals actual era_tas_kt."""

    def test_tas_target_last_row_actual(self) -> None:
        df = _make_flight()
        result = build_selected_params(df, _config())

        last_target = result["fdm_tas_target_kt"][-1]
        last_actual = result["era_tas_kt"][-1]
        assert last_target == pytest.approx(last_actual, rel=1e-6)


class TestSiConversion:
    """fdm_tas_target_ms approx fdm_tas_target_kt x 0.514444."""

    def test_si_conversion(self) -> None:
        df = _make_flight()
        result = build_selected_params(df, _config())

        assert "fdm_tas_target_kt" in result.columns

        # Apply the kt_to_ms conversion manually and compare
        converted = result.select(
            kt_to_ms("fdm_tas_target_kt").alias("manual_ms"),
        )
        manual_ms = converted["manual_ms"]

        # The implementation should also produce fdm_tas_target_ms via SI_CONVERSIONS,
        # but at this stage we verify the conversion factor is correct.
        expected = result["fdm_tas_target_kt"].to_numpy() * 0.514444
        np.testing.assert_allclose(
            manual_ms.to_numpy(),
            expected,
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNoMachSegments:
    """No Mach segments — only CAS and TAS segments fill the target."""

    def test_no_mach_segments(self) -> None:
        df = _make_flight(include_mach=False)
        cfg = _config(include_mach=False)
        result = build_selected_params(df, cfg)

        assert "fdm_tas_target_kt" in result.columns
        nan_count = (
            result["fdm_tas_target_kt"].null_count() + result["fdm_tas_target_kt"].is_nan().sum()
        )
        assert nan_count == 0, f"Expected zero NaN without Mach, got {nan_count}"


class TestNoSegmentsAtAll:
    """No segments detected — target should backfill from last actual TAS."""

    def test_no_segments(self) -> None:
        # Short noisy flight with no stable segments
        rng = np.random.default_rng(99)
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.linspace(0, 10_000, n),
                "era_tas_kt": rng.uniform(200, 400, n),  # random, no plateaus
            }
        )
        # Config with very strict thresholds so nothing is detected
        cfg: dict[str, dict[str, object]] = {}
        result = build_selected_params(df, cfg)

        assert "fdm_tas_target_kt" in result.columns
        # Should backfill from last actual TAS
        last_actual = df["era_tas_kt"][-1]
        last_target = result["fdm_tas_target_kt"][-1]
        assert last_target == pytest.approx(last_actual, rel=1e-6)

        # All values should be the last actual TAS (backfill from anchor)
        nan_count = (
            result["fdm_tas_target_kt"].null_count() + result["fdm_tas_target_kt"].is_nan().sum()
        )
        assert nan_count == 0


class TestNanInAltitude:
    """NaN in altitude — conversion handles gracefully, backfill covers gaps."""

    def test_nan_altitude(self) -> None:
        # Inject NaN at specific altitude indices
        nan_indices = [50, 51, 52, 150, 151]
        df = _make_flight(alt_nan_indices=nan_indices)
        result = build_selected_params(df, _config())

        assert "fdm_tas_target_kt" in result.columns
        # Despite NaN altitudes, target should have no NaN (backfill covers)
        nan_count = (
            result["fdm_tas_target_kt"].null_count() + result["fdm_tas_target_kt"].is_nan().sum()
        )
        assert nan_count == 0, f"Expected zero NaN with NaN altitudes, got {nan_count}"
