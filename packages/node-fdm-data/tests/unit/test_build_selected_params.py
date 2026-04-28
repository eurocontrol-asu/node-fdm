"""TDD tests for crossover-aware Mach/CAS segment detection in build_selected_params.

Verifies the public contract of node_fdm_data.segments.build_selected_params
for AC1-AC8 of AXM-1625. Tests exercise the public boundary only — no
private helpers (_optimize_transition_cas, _build_tas_target, etc.) are
imported directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.segments import build_selected_params

_FT_TO_M = 0.3048
_KT_TO_MS = 0.5144444444
_MS_TO_KT = 1.0 / _KT_TO_MS


def _config(**overrides: dict) -> dict:
    cfg: dict = {
        "mach": {"min_length": 30, "tolerance": 0.005},
        "cas": {"min_length": 30, "tolerance": 2.0},
        "vz": {"min_length": 10, "tolerance": 50.0},
        "alt": {"min_length": 30, "tolerance": 50.0},
    }
    cfg.update(overrides)
    return cfg


def _flight(  # noqa: PLR0913
    *,
    alt_ft: np.ndarray,
    mach: np.ndarray,
    cas: np.ndarray,
    vz: np.ndarray | None = None,
    tas_kt: np.ndarray | None = None,
    era_temp_K: np.ndarray | None = None,  # noqa: N803
) -> pl.DataFrame:
    n = alt_ft.size
    if vz is None:
        vz = np.zeros(n)
    cols: dict = {
        "era_mach": mach,
        "bds_ias_kt": cas,
        "raw_alt_ft": alt_ft,
        "raw_vz_ftmin": vz,
    }
    if tas_kt is not None:
        cols["era_tas_kt"] = tas_kt
    if era_temp_K is not None:
        cols["era_temp_K"] = era_temp_K
    return pl.DataFrame(cols)


def _three_phase_alt(
    n_climb: int, n_cruise: int, n_descent: int, cruise_ft: float = 34000.0
) -> np.ndarray:
    n = n_climb + n_cruise + n_descent
    alt = np.empty(n, dtype=np.float64)
    alt[:n_climb] = np.linspace(1000.0, cruise_ft, n_climb)
    alt[n_climb : n_climb + n_cruise] = cruise_ft
    alt[n_climb + n_cruise :] = np.linspace(cruise_ft, 1000.0, n_descent)
    return alt


class TestMachPlateauOnly:
    """AC1: Mach detected only on altitude plateaus."""

    def test_mach_detected_only_on_alt_plateau(self):
        n_climb, n_cruise, n_descent = 100, 100, 100
        alt = _three_phase_alt(n_climb, n_cruise, n_descent)
        n = alt.size
        mach = np.full(n, 0.78)  # constant Mach across full flight
        cas = np.full(n, np.nan)

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())

        sel = out["fdm_mach_sel"].to_numpy()
        plateau_mask = alt == 34000.0
        # Some plateau rows must have non-NaN Mach
        assert np.any(~np.isnan(sel[plateau_mask]))
        # All non-plateau rows must be NaN
        assert np.all(np.isnan(sel[~plateau_mask]))

    def test_mach_outside_plateau_dropped(self):
        # Climb-only — no altitude plateau
        n = 200
        alt = np.linspace(1000.0, 34000.0, n)
        mach = np.full(n, 0.78)
        cas = np.full(n, np.nan)

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())

        assert np.all(np.isnan(out["fdm_mach_sel"].to_numpy()))


class TestAberrantMachFilter:
    """AC7: Reject Mach segments with mean < 0.5."""

    def test_aberrant_low_mach_segment_filtered(self):
        alt = _three_phase_alt(100, 100, 100)
        n = alt.size
        mach = np.full(n, 0.23)  # ghost-data value
        cas = np.full(n, np.nan)

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())

        assert np.all(np.isnan(out["fdm_mach_sel"].to_numpy()))

    def test_aberrant_high_mach_segment_kept(self):
        alt = _three_phase_alt(100, 100, 100)
        n = alt.size
        mach = np.full(n, 0.78)
        cas = np.full(n, np.nan)

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())

        sel = out["fdm_mach_sel"].to_numpy()
        plateau_mask = alt == 34000.0
        assert np.any(~np.isnan(sel[plateau_mask]))


class TestCasOptimisation:
    """AC2/AC3: Transition CAS optimisation."""

    @staticmethod
    def _build_consistent_flight(  # noqa: PLR0913
        *,
        cas_climb_kt: float,
        cas_descent_kt: float,
        mach_cruise: float,
        n_climb: int = 100,
        n_cruise: int = 100,
        n_descent: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas

        n = n_climb + n_cruise + n_descent
        alt_ft = _three_phase_alt(n_climb, n_cruise, n_descent)
        alt_m = alt_ft * _FT_TO_M

        mach = np.full(n, np.nan)
        mach[n_climb : n_climb + n_cruise] = mach_cruise

        cas_real_kt = np.full(n, np.nan)
        cas_real_kt[:n_climb] = cas_climb_kt
        cas_real_kt[n_climb + n_cruise :] = cas_descent_kt

        tas_ms = np.full(n, np.nan)
        tas_ms[:n_climb] = np.asarray(cas_to_tas(cas_climb_kt * _KT_TO_MS, alt_m[:n_climb]))
        tas_ms[n_climb : n_climb + n_cruise] = np.asarray(
            mach_to_tas(mach_cruise, alt_m[n_climb : n_climb + n_cruise])
        )
        tas_ms[n_climb + n_cruise :] = np.asarray(
            cas_to_tas(cas_descent_kt * _KT_TO_MS, alt_m[n_climb + n_cruise :])
        )
        tas_real_kt = tas_ms * _MS_TO_KT
        return alt_ft, mach, cas_real_kt, tas_real_kt

    def test_cas_optimisation_recovers_known_value(self):
        alt_ft, mach, cas_real_kt, tas_kt = self._build_consistent_flight(
            cas_climb_kt=280.0,
            cas_descent_kt=270.0,
            mach_cruise=0.78,
        )

        out = build_selected_params(
            _flight(alt_ft=alt_ft, mach=mach, cas=cas_real_kt, tas_kt=tas_kt),
            _config(),
        )
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()

        climb_vals = cas_sel[:100]
        non_nan = climb_vals[~np.isnan(climb_vals)]
        assert non_nan.size > 0
        # Optimised CAS should be within 1 kt of 280 over the climb window
        assert np.all(np.abs(non_nan - 280.0) <= 1.0)

    def test_cas_deviation_cutoff(self):
        alt_ft, mach, cas_real_kt, tas_kt = self._build_consistent_flight(
            cas_climb_kt=280.0,
            cas_descent_kt=270.0,
            mach_cruise=0.78,
        )
        # Inject 8-kt CAS deviation in early climb (rows 0..49)
        cas_real_kt = cas_real_kt.copy()
        cas_real_kt[:50] = 288.0  # deviates by 8 kt from optimised 280

        out = build_selected_params(
            _flight(alt_ft=alt_ft, mach=mach, cas=cas_real_kt, tas_kt=tas_kt),
            _config(),
        )
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        # Rows where real CAS deviates >5 kt should stay NaN
        assert np.all(np.isnan(cas_sel[:50]))


class TestEraTemperature:
    """AC4: Use era_temp_K when present, fall back to ISA otherwise."""

    def test_uses_era_temp_when_present(self):
        from node_fdm_data.physics.speed import isa_temperature

        alt = _three_phase_alt(100, 100, 100)
        alt_m = alt * _FT_TO_M
        n = alt.size
        mach = np.full(n, np.nan)
        mach[100:200] = 0.78
        cas = np.full(n, np.nan)
        # ISA minus 10 K everywhere
        temp = np.asarray(isa_temperature(alt_m), dtype=np.float64) - 10.0

        out = build_selected_params(
            _flight(alt_ft=alt, mach=mach, cas=cas, era_temp_K=temp),
            _config(),
        )

        # Real-temp variant must exist after impl
        from node_fdm_data.physics.speed import mach_to_tas_real

        expected_ms = np.asarray(mach_to_tas_real(0.78, temp[100:200]))
        expected_kt = expected_ms * _MS_TO_KT
        target = out["fdm_tas_target_kt"].to_numpy()
        idx = np.where(~np.isnan(target[100:200]))[0]
        assert idx.size > 0
        for i in idx[:5]:
            assert abs(target[100 + i] - expected_kt[i]) < 1.0

    def test_falls_back_to_isa_when_era_temp_absent(self):
        from node_fdm_data.physics.speed import mach_to_tas

        alt = _three_phase_alt(100, 100, 100)
        alt_m = alt * _FT_TO_M
        n = alt.size
        mach = np.full(n, np.nan)
        mach[100:200] = 0.78
        cas = np.full(n, np.nan)

        out = build_selected_params(
            _flight(alt_ft=alt, mach=mach, cas=cas),  # no era_temp_K
            _config(),
        )

        expected_kt = np.asarray(mach_to_tas(0.78, alt_m[100:200])) * _MS_TO_KT
        target = out["fdm_tas_target_kt"].to_numpy()
        idx = np.where(~np.isnan(target[100:200]))[0]
        assert idx.size > 0
        for i in idx[:5]:
            assert abs(target[100 + i] - expected_kt[i]) < 0.5


class TestTasTargetEnvelope:
    """AC5: TAS target = min(mach_to_tas, cas_to_tas) on overlap; NaN outside."""

    def test_tas_target_uses_envelope_on_overlap(self):
        from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas

        alt_ft, mach, cas_real_kt, tas_kt = TestCasOptimisation._build_consistent_flight(
            cas_climb_kt=280.0,
            cas_descent_kt=270.0,
            mach_cruise=0.78,
        )
        alt_m = alt_ft * _FT_TO_M

        out = build_selected_params(
            _flight(alt_ft=alt_ft, mach=mach, cas=cas_real_kt, tas_kt=tas_kt),
            _config(),
        )
        mach_sel = out["fdm_mach_sel"].to_numpy()
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        target = out["fdm_tas_target_kt"].to_numpy()

        overlap = ~np.isnan(mach_sel) & ~np.isnan(cas_sel)
        if not overlap.any():
            pytest.skip("synthetic flight produced no overlap rows")
        tas_mach = np.asarray(mach_to_tas(mach_sel[overlap], alt_m[overlap])) * _MS_TO_KT
        tas_cas = np.asarray(cas_to_tas(cas_sel[overlap] * _KT_TO_MS, alt_m[overlap])) * _MS_TO_KT
        expected = np.minimum(tas_mach, tas_cas)
        np.testing.assert_allclose(target[overlap], expected, atol=1.0)

    def test_tas_target_nan_outside_segments(self):
        # No plateau, no CAS plateau → no segments at all
        n = 200
        alt = np.linspace(1000.0, 34000.0, n)
        mach = np.full(n, np.nan)
        cas = np.full(n, np.nan)

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())

        assert np.all(np.isnan(out["fdm_tas_target_kt"].to_numpy()))

    def test_tas_target_no_global_backfill(self):
        # Flight ends with 50 NaN rows after the last segment
        n_climb, n_cruise, n_tail = 100, 100, 50
        n = n_climb + n_cruise + n_tail
        alt = np.empty(n)
        alt[:n_climb] = np.linspace(1000.0, 34000.0, n_climb)
        alt[n_climb : n_climb + n_cruise] = 34000.0
        alt[n_climb + n_cruise :] = np.nan
        mach = np.full(n, np.nan)
        mach[n_climb : n_climb + n_cruise] = 0.78
        cas = np.full(n, np.nan)

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())
        target = out["fdm_tas_target_kt"].to_numpy()
        # Last 50 rows must remain NaN (no backward-fill from earlier cruise)
        assert np.all(np.isnan(target[-n_tail:]))


class TestTasTargetKnownMask:
    """AC6: fdm_tas_target_known column."""

    def test_tas_target_known_mask_emitted(self):
        alt = _three_phase_alt(100, 100, 100)
        n = alt.size
        mach = np.full(n, np.nan)
        mach[100:200] = 0.78
        cas = np.full(n, np.nan)

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())

        assert "fdm_tas_target_known" in out.columns
        known = out["fdm_tas_target_known"].to_numpy()
        target = out["fdm_tas_target_kt"].to_numpy()
        np.testing.assert_array_equal(known.astype(bool), ~np.isnan(target))


class TestNoOptimisationEdgeCases:
    """AC8: Skip optimisation when first/last plateau is within margin."""

    def test_no_climb_skips_optimisation(self):
        # First Mach plateau starts at row 5 < default margin (30)
        n = 200
        alt = np.empty(n)
        alt[:5] = np.linspace(33000.0, 34000.0, 5)
        alt[5:150] = 34000.0
        alt[150:] = np.linspace(34000.0, 1000.0, 50)
        mach = np.full(n, np.nan)
        mach[5:150] = 0.78
        cas = np.full(n, np.nan)
        cas[150:] = 270.0

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        # No optimised CAS in (very short) climb — must not raise
        assert np.all(np.isnan(cas_sel[:5]))

    def test_no_descent_skips_optimisation(self):
        # Last Mach plateau ends at row n-5 < default margin (30)
        n = 200
        alt = np.empty(n)
        alt[:50] = np.linspace(1000.0, 34000.0, 50)
        alt[50:195] = 34000.0
        alt[195:] = np.linspace(34000.0, 33000.0, 5)
        mach = np.full(n, np.nan)
        mach[50:195] = 0.78
        cas = np.full(n, np.nan)
        cas[:50] = 280.0

        out = build_selected_params(_flight(alt_ft=alt, mach=mach, cas=cas), _config())
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        # No optimised CAS in (very short) descent — must not raise
        assert np.all(np.isnan(cas_sel[-5:]))


class TestEmptyDataFrame:
    """AC1: Empty DataFrame — no error, expected columns present."""

    def test_empty_dataframe(self):
        df = pl.DataFrame(
            {
                "era_mach": pl.Series([], dtype=pl.Float64),
                "bds_ias_kt": pl.Series([], dtype=pl.Float64),
                "raw_alt_ft": pl.Series([], dtype=pl.Float64),
                "raw_vz_ftmin": pl.Series([], dtype=pl.Float64),
            }
        )
        out = build_selected_params(df, _config())
        for col in (
            "fdm_mach_sel",
            "fdm_cas_sel_kt",
            "fdm_alt_sel_ft",
            "fdm_tas_target_kt",
            "fdm_tas_target_known",
        ):
            assert col in out.columns
        assert len(out) == 0
