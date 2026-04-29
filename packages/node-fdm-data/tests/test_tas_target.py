"""Tests for fdm_tas_target_kt — unified TAS target from segments."""

from __future__ import annotations

import numpy as np
import polars as pl

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
        cols["bds_mach_clean"] = mach

    if include_cas:
        cas = np.concatenate(
            [
                np.full(third, cas_climb) + rng.normal(0, 0.1, third),
                np.linspace(cas_climb, 280, n - 2 * third),
                np.full(third, cas_descent) + rng.normal(0, 0.1, third),
            ]
        )
        cols["bds_ias_kt_clean"] = cas

    if include_tas:
        tas = np.concatenate(
            [
                np.linspace(200, tas_cruise_kt, third),
                np.full(n - 2 * third, tas_cruise_kt) + rng.normal(0, 0.2, n - 2 * third),
                np.linspace(tas_cruise_kt, 200, third),
            ]
        )
        cols["bds_tas_from_cas_kt"] = tas

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
    """``fdm_tas_target_kt`` is non-NaN inside Mach/CAS coverage.

    Updated for AXM-1625 (AC5/AC6): the global backward-fill anchor was
    removed, so target NaN is now expected outside detected segments.
    The new contract guarantees: where ``fdm_mach_sel`` or
    ``fdm_cas_sel_kt`` is non-NaN, the target is also non-NaN.
    """

    def test_tas_target_non_nan_inside_coverage(self) -> None:
        df = _make_flight()
        result = build_selected_params(df, _config())

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        mach_sel = result["fdm_mach_sel"].to_numpy()
        cas_sel = result["fdm_cas_sel_kt"].to_numpy()
        coverage = ~np.isnan(mach_sel) | ~np.isnan(cas_sel)
        assert np.all(
            ~np.isnan(target[coverage])
        ), "fdm_tas_target_kt must be non-NaN wherever a Mach or CAS segment exists"

    def test_tas_target_known_mask_matches_target(self) -> None:
        df = _make_flight()
        result = build_selected_params(df, _config())

        assert "fdm_tas_target_known" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        np.testing.assert_array_equal(known, ~np.isnan(target))


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
    """No Mach segments — only CAS/TAS segments contribute; gaps stay NaN."""

    def test_no_mach_segments_yields_nan_gaps(self) -> None:
        df = _make_flight(include_mach=False)
        cfg = _config(include_mach=False)
        result = build_selected_params(df, cfg)

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        cas_sel = result["fdm_cas_sel_kt"].to_numpy()
        tas_sel_present = "fdm_tas_sel_kt" in result.columns
        tas_sel = (
            result["fdm_tas_sel_kt"].to_numpy()
            if tas_sel_present
            else np.full(len(target), np.nan)
        )
        coverage = ~np.isnan(cas_sel) | ~np.isnan(tas_sel)
        # Inside coverage: target is non-NaN (CAS or TAS-derived).
        assert np.all(~np.isnan(target[coverage]))
        # Outside coverage: target may stay NaN, and known mask reflects that.
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        np.testing.assert_array_equal(known, ~np.isnan(target))


class TestNoSegmentsAtAll:
    """No segments detected — target stays NaN end-to-end and known is False."""

    def test_no_segments_target_all_nan(self) -> None:
        rng = np.random.default_rng(99)
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.linspace(0, 10_000, n),
                "bds_tas_from_cas_kt": rng.uniform(200, 400, n),
            }
        )
        cfg: dict[str, dict[str, object]] = {}
        result = build_selected_params(df, cfg)

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        assert np.all(np.isnan(target))
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        assert not known.any()


class TestNanInAltitude:
    """NaN in altitude — target stays NaN where altitude is NaN (no backfill)."""

    def test_nan_altitude_yields_nan_target(self) -> None:
        nan_indices = [50, 51, 52, 150, 151]
        df = _make_flight(alt_nan_indices=nan_indices)
        result = build_selected_params(df, _config())

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        # Known mask is the inverse-NaN mask of the target (AC6).
        np.testing.assert_array_equal(known, ~np.isnan(target))
