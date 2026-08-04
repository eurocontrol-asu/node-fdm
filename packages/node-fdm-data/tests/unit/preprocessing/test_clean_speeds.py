"""Tests for speed cleaning (Hampel + ERA5 fill)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds, clean_speeds


class TestHampel:
    """Hampel filter behaviour."""

    def test_hampel_removes_isolated_spike(self) -> None:
        """A single 10-sigma spike is replaced by NaN; neighbours unchanged."""
        n = 50
        values = np.full(n, 100.0)
        values[25] = 1000.0

        result = clean_speeds(
            values,
            window=7,
            k=3.0,
            n_passes=1,
            interp_max_gap=0,
        )

        assert math.isnan(result[25])
        assert result[24] == pytest.approx(100.0)
        assert result[26] == pytest.approx(100.0)

    def test_hampel_keeps_clean_signal(self) -> None:
        """A smooth sinusoid passes through unchanged (no false positives)."""
        n = 100
        t = np.arange(n)
        values = 100.0 + 5.0 * np.sin(t * 0.1)

        result = clean_speeds(
            values.copy(),
            window=7,
            k=3.0,
            n_passes=3,
            interp_max_gap=0,
        )

        np.testing.assert_allclose(result, values, equal_nan=True)

    def test_hampel_multi_pass_removes_cluster(self) -> None:
        """A 3-point outlier cluster is fully removed via multi-pass Hampel."""
        n = 50
        values = np.full(n, 100.0)
        values[24] = 1000.0
        values[25] = 1000.0
        values[26] = 1000.0

        result = clean_speeds(
            values,
            window=7,
            k=3.0,
            n_passes=3,
            interp_max_gap=0,
        )

        assert math.isnan(result[24])
        assert math.isnan(result[25])
        assert math.isnan(result[26])


class TestInterpolation:
    """Linear interpolation of short gaps."""

    def test_interpolate_short_gap(self) -> None:
        """A NaN run of length 3 between valid anchors is filled linearly."""
        values = np.array([10.0, 11.0, np.nan, np.nan, np.nan, 15.0, 16.0])

        result = clean_speeds(
            values,
            window=3,
            k=10.0,
            n_passes=1,
            interp_max_gap=10,
        )

        assert result[2] == pytest.approx(12.0)
        assert result[3] == pytest.approx(13.0)
        assert result[4] == pytest.approx(14.0)

    def test_interpolate_leaves_long_gap(self) -> None:
        """A NaN run of length 20 with interp_max_gap=10 stays NaN."""
        values = np.full(30, np.nan)
        values[0] = 1.0
        values[29] = 100.0

        result = clean_speeds(
            values,
            window=3,
            k=10.0,
            n_passes=1,
            interp_max_gap=10,
        )

        for i in range(1, 29):
            assert math.isnan(result[i])

    def test_interpolate_gap_at_edges(self) -> None:
        """NaN run with no left or right anchor stays NaN (no extrapolation)."""
        n = 20
        values = np.full(n, np.nan)
        values[8:12] = 100.0

        result = clean_speeds(
            values,
            window=3,
            k=10.0,
            n_passes=1,
            interp_max_gap=20,
        )

        for i in range(0, 8):
            assert math.isnan(result[i])
        for i in range(12, n):
            assert math.isnan(result[i])


class TestCleanSpeedsAllNan:
    """Edge case: all-NaN input."""

    def test_clean_speeds_all_nan_input(self) -> None:
        """All-NaN values returns all-NaN, no error raised."""
        n = 30
        values = np.full(n, np.nan)

        result = clean_speeds(
            values,
            window=7,
            k=3.0,
            n_passes=3,
            interp_max_gap=10,
        )

        assert all(math.isnan(v) for v in result)


class TestCleanBdsSpeeds:
    """Polars wrapper clean_bds_speeds."""

    @staticmethod
    def _make_df() -> pl.DataFrame:
        n = 50
        # Small jitter rather than a constant. A strictly constant run is what
        # `_flag_frozen_runs` exists to delete — a real BDS signal never repeats
        # to the fourth decimal for 50 samples — and with the ERA5 fill now off
        # for Mach and CAS there is nothing left to resurrect the column.
        rng = np.random.default_rng(0)
        bds_mach = 0.80 + rng.normal(0, 2e-4, n)
        bds_ias_kt = 250.0 + rng.normal(0, 0.3, n)
        bds_tas_kt = 230.0 + rng.normal(0, 0.3, n)
        bds_mach[25] = 1.5
        bds_ias_kt[25] = 800.0
        bds_tas_kt[25] = 800.0

        # alt > on_ground_alt_threshold so the on-ground mask never triggers
        # in this fixture; it is exercised separately in TestOnGroundMask.
        return pl.DataFrame(
            {
                "bds_mach": bds_mach,
                "bds_ias_kt": bds_ias_kt,
                "bds_tas_kt": bds_tas_kt,
                "era_mach": np.full(n, 0.80),
                "era_tas_kt": np.full(n, 460.0),
                "era_cas_kt": np.full(n, 250.0),
                "era_temp_K": np.full(n, 220.0),
                "raw_alt_ft": np.full(n, 35000.0),
                "raw_vz_ftmin": np.full(n, 0.0),
            }
        )

    def test_clean_bds_speeds_creates_clean_columns(self) -> None:
        df = self._make_df()
        result = clean_bds_speeds(df)
        assert "bds_mach_clean" in result.columns
        assert "bds_ias_kt_clean" in result.columns
        assert "bds_tas_kt_clean" in result.columns

    def test_clean_bds_speeds_preserves_raw(self) -> None:
        df = self._make_df()
        raw_mach = df["bds_mach"].to_list()
        raw_ias = df["bds_ias_kt"].to_list()
        raw_tas = df["bds_tas_kt"].to_list()

        result = clean_bds_speeds(df)

        assert result["bds_mach"].to_list() == raw_mach
        assert result["bds_ias_kt"].to_list() == raw_ias
        assert result["bds_tas_kt"].to_list() == raw_tas

    def test_clean_bds_speeds_idempotent(self) -> None:
        df = self._make_df()
        first = clean_bds_speeds(df)
        second = clean_bds_speeds(first)

        for col in ("bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean"):
            a = first[col].to_list()
            b = second[col].to_list()
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=False):
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    assert y is None or (isinstance(y, float) and math.isnan(y))
                else:
                    assert x == pytest.approx(y)

    def test_clean_bds_speeds_missing_input_columns(self) -> None:
        """When bds_mach is absent, clean function must not raise."""
        df = pl.DataFrame(
            {
                "bds_ias_kt": [250.0, 251.0, 252.0],
                "bds_tas_kt": [230.0, 231.0, 232.0],
                "era_tas_kt": [460.0, 461.0, 462.0],
                "era_cas_kt": [250.0, 251.0, 252.0],
            }
        )
        result = clean_bds_speeds(df)
        if "bds_mach_clean" in result.columns:
            assert result["bds_mach_clean"].null_count() == result.height

    def test_clean_bds_speeds_derives_tas_from_cas(self) -> None:
        """``fdm_tas_from_cas_kt`` is appended when alt+temp are present."""
        df = self._make_df()
        result = clean_bds_speeds(df)
        assert "fdm_tas_from_cas_kt" in result.columns
        # At FL350 with T=220K and CAS=250 kt, TAS should be ~410-450 kt.
        tas = result["fdm_tas_from_cas_kt"].to_numpy()
        finite = tas[~np.isnan(tas)]
        assert finite.size > 0
        assert finite.min() > 350.0
        assert finite.max() < 500.0

    def test_clean_bds_speeds_skips_tas_derivation_without_meteo(self) -> None:
        """No ``fdm_tas_from_cas_kt`` column when era_temp_K / raw_alt_ft missing."""
        n = 30
        df = pl.DataFrame(
            {
                "bds_mach": 0.80 + np.random.default_rng(1).normal(0, 2e-4, n),
                "bds_ias_kt": 250.0 + np.random.default_rng(2).normal(0, 0.3, n),
                "bds_tas_kt": np.full(n, 230.0),
                "era_mach": np.full(n, 0.80),
                "era_cas_kt": np.full(n, 250.0),
                "era_tas_kt": np.full(n, 460.0),
                # no raw_alt_ft, no era_temp_K -> derivation should skip
            }
        )
        result = clean_bds_speeds(df)
        assert "fdm_tas_from_cas_kt" not in result.columns


class TestPointJumps:
    """V-shape point-jump filter behaviour."""

    def test_v_shape_spike_is_flagged(self) -> None:
        """A single isolated up-then-down spike is flagged as NaN."""
        n = 30
        values = np.full(n, 100.0)
        values[15] = 200.0  # up by 100, then down by 100 → V-shape

        result = clean_speeds(
            values,
            window=7,
            k=3.0,
            n_passes=1,
            interp_max_gap=0,
            point_jump_max=50.0,
        )
        assert math.isnan(result[15])

    def test_gradient_is_kept(self) -> None:
        """A monotonic ramp (no V-shape) is preserved."""
        n = 30
        values = 100.0 + np.arange(n, dtype=np.float64) * 10.0  # ramp +10/sample

        result = clean_speeds(
            values.copy(),
            window=7,
            k=10.0,  # disable Hampel
            n_passes=0,  # no Hampel
            interp_max_gap=0,
            point_jump_max=5.0,  # smaller than the ramp slope
        )
        # No point should be flagged: deltas have the same sign (all positive).
        assert not np.any(np.isnan(result))

    def test_disabled_when_none(self) -> None:
        """``point_jump_max=None`` is a no-op."""
        n = 30
        values = np.full(n, 100.0)
        values[15] = 1000.0  # huge V-shape

        result = clean_speeds(
            values.copy(),
            window=7,
            k=10.0,
            n_passes=0,
            interp_max_gap=0,
            point_jump_max=None,
        )
        assert result[15] == pytest.approx(1000.0)


class TestZigzagRegion:
    """Zigzag-region detector behaviour."""

    def test_dense_zigzag_zone_is_flagged(self) -> None:
        """A dense zigzag region is NaN-ed out as a block."""
        n = 80
        values = np.full(n, 100.0)
        # 30-point zigzag in the middle: alternating 100 and 200
        for i in range(25, 55):
            values[i] = 100.0 if (i % 2 == 0) else 200.0

        result = clean_speeds(
            values.copy(),
            window=7,
            k=10.0,  # disable Hampel
            n_passes=0,
            interp_max_gap=0,
            zigzag_jump_min=50.0,
            zigzag_half_window=10,
            zigzag_density_min=0.4,
        )
        # Most of the zigzag block should be NaN (centre fully covered).
        assert np.isnan(result[35])
        assert np.isnan(result[40])

    def test_smooth_signal_is_kept(self) -> None:
        """A smooth signal with small deltas is preserved."""
        n = 50
        values = 100.0 + np.arange(n, dtype=np.float64) * 0.1

        result = clean_speeds(
            values.copy(),
            window=7,
            k=10.0,
            n_passes=0,
            interp_max_gap=0,
            zigzag_jump_min=10.0,  # threshold much larger than 0.1
            zigzag_half_window=10,
            zigzag_density_min=0.3,
        )
        assert not np.any(np.isnan(result))

    def test_disabled_when_none(self) -> None:
        """``zigzag_jump_min=None`` is a no-op."""
        n = 50
        values = np.full(n, 100.0)
        for i in range(20, 40):  # 20-point zigzag
            values[i] = 100.0 if (i % 2 == 0) else 200.0

        result = clean_speeds(
            values.copy(),
            window=7,
            k=10.0,
            n_passes=0,
            interp_max_gap=0,
            zigzag_jump_min=None,
        )
        assert not np.any(np.isnan(result))


class TestOnGroundMask:
    """``clean_bds_speeds`` applies the on-ground mask to outputs."""

    @staticmethod
    def _make_df_with_ground_segments(n: int = 60) -> pl.DataFrame:
        """First 10 rows on ground (alt=600, vz=0), then airborne (alt=35k, vz=0).

        BDS / ERA values jitter slightly to avoid the frozen-run filter
        flagging the whole fixture as a stuck-sensor plateau.
        """
        rng = np.random.default_rng(0)
        bds_mach = 0.80 + rng.normal(0.0, 0.001, n)
        bds_ias_kt = 250.0 + rng.normal(0.0, 0.5, n)
        bds_tas_kt = 230.0 + rng.normal(0.0, 0.5, n)
        alt = np.full(n, 35000.0)
        alt[:10] = 600.0  # on ground
        vz = np.full(n, 0.0)
        return pl.DataFrame(
            {
                "bds_mach": bds_mach,
                "bds_ias_kt": bds_ias_kt,
                "bds_tas_kt": bds_tas_kt,
                "era_mach": 0.80 + rng.normal(0.0, 0.001, n),
                "era_cas_kt": 250.0 + rng.normal(0.0, 0.5, n),
                "era_tas_kt": 460.0 + rng.normal(0.0, 0.5, n),
                "era_temp_K": np.full(n, 220.0),
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
            }
        )

    def test_on_ground_rows_are_nan_in_clean_output(self) -> None:
        """Cleaned rows where alt<1500 and |vz|<200 are forced to NaN."""
        df = self._make_df_with_ground_segments()
        result = clean_bds_speeds(df)
        for col in ("bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean"):
            arr = result[col].to_numpy()
            # First 10 rows are on ground -> all NaN
            assert np.all(np.isnan(arr[:10]))
            # Airborne rows -> at least some finite
            assert np.any(~np.isnan(arr[10:]))

    def test_airborne_high_vz_keeps_clean_value(self) -> None:
        """A point with low alt but high |vz| (taking off) is NOT on ground."""
        n = 30
        # Jittered rather than constant: `_flag_frozen_runs` deletes strictly
        # identical runs, and with the ERA5 fill now off for Mach and CAS there
        # is nothing to put the column back. The jitter is far below what this
        # test is about — whether the on-ground mask fires — so the assertion
        # still measures only that.
        rng = np.random.default_rng(3)
        df = pl.DataFrame(
            {
                "bds_mach": 0.50 + rng.normal(0, 2e-4, n),
                "bds_ias_kt": 200.0 + rng.normal(0, 0.3, n),
                "bds_tas_kt": 200.0 + rng.normal(0, 0.3, n),
                "era_mach": np.full(n, 0.50),
                "era_cas_kt": np.full(n, 200.0),
                "era_tas_kt": np.full(n, 250.0),
                "era_temp_K": np.full(n, 250.0),
                "raw_alt_ft": np.full(n, 1000.0),  # below alt threshold
                "raw_vz_ftmin": np.full(n, 2500.0),  # but climbing -> airborne
            }
        )
        result = clean_bds_speeds(df)
        ias = result["bds_ias_kt_clean"].to_numpy()
        # Mostly finite (climbing -> not on ground)
        assert np.sum(~np.isnan(ias)) > n // 2

    def test_no_mask_when_columns_missing(self) -> None:
        """Without raw_vz_ftmin / raw_alt_ft the mask is skipped."""
        n = 30
        df = pl.DataFrame(
            {
                "bds_mach": 0.80 + np.random.default_rng(1).normal(0, 2e-4, n),
                "bds_ias_kt": 250.0 + np.random.default_rng(2).normal(0, 0.3, n),
                "bds_tas_kt": np.full(n, 230.0),
                "era_mach": np.full(n, 0.80),
                "era_cas_kt": np.full(n, 250.0),
                "era_tas_kt": np.full(n, 460.0),
            }
        )
        result = clean_bds_speeds(df)
        # No on-ground mask applied -> finite values survive
        ias = result["bds_ias_kt_clean"].to_numpy()
        assert np.sum(~np.isnan(ias)) > n // 2


class TestFrozenRunFilter:
    """Frozen-signal filter behaviour."""

    def test_frozen_run_at_threshold_is_nan(self) -> None:
        """A run of exactly min_run_len identical values is flagged as NaN."""
        n = 50
        values = np.full(n, 100.0)
        values[10:30] = 200.0  # run of length 20

        result = clean_speeds(
            values,
            window=7,
            k=3.0,
            n_passes=1,
            interp_max_gap=0,
            frozen_min_run_len=20,
        )

        assert all(math.isnan(v) for v in result[10:30])

    def test_frozen_run_below_threshold_is_kept(self) -> None:
        """A run of length below min_run_len is preserved."""
        n = 50
        values = np.full(n, 100.0)
        values[10:29] = 200.0  # run of length 19

        result = clean_speeds(
            values,
            window=7,
            k=3.0,
            n_passes=0,
            interp_max_gap=0,
            frozen_min_run_len=20,
        )

        for v in result[10:29]:
            assert v == pytest.approx(200.0)

    def test_short_cruise_plateau_is_kept(self) -> None:
        """A 5-point identical cruise plateau is preserved with high threshold."""
        n = 50
        values = np.arange(n, dtype=np.float64) * 0.5
        values[20:25] = 250.0  # legitimate 5-point plateau

        result = clean_speeds(
            values.copy(),
            window=7,
            k=3.0,
            n_passes=0,
            interp_max_gap=0,
            frozen_min_run_len=20,
        )

        for v in result[20:25]:
            assert v == pytest.approx(250.0)

    def test_disabled_when_none(self) -> None:
        """frozen_min_run_len=None is the no-op (backward compatibility)."""
        n = 50
        values = np.full(n, 100.0)  # whole array identical, length 50

        result = clean_speeds(
            values.copy(),
            window=7,
            k=3.0,
            n_passes=0,
            interp_max_gap=0,
            frozen_min_run_len=None,
        )

        np.testing.assert_array_equal(result, values)

    def test_nan_breaks_run(self) -> None:
        """NaN in the middle splits a run into two sub-runs, neither flagged."""
        n = 50
        values = np.full(n, 100.0)
        values[10:30] = 200.0
        values[20] = np.nan  # splits run-of-20 into 10 + 9

        result = clean_speeds(
            values,
            window=7,
            k=3.0,
            n_passes=0,
            interp_max_gap=0,
            frozen_min_run_len=20,
        )

        # Both sub-runs preserved
        for v in result[10:20]:
            assert v == pytest.approx(200.0)
        for v in result[21:30]:
            assert v == pytest.approx(200.0)
        assert math.isnan(result[20])

    def test_clean_bds_speeds_applies_per_channel_thresholds(self) -> None:
        """clean_bds_speeds wires per-channel frozen thresholds correctly.

        Same input on both channels: 6-point identical leading run + smooth
        ramp.  With ``frozen_min_run_len_tas=6`` the run is flagged as NaN
        on tas; with ``frozen_min_run_len_ias=20`` the same run is kept on
        ias (below threshold).  ``n_passes=0`` and ``interp_max_gap=0``
        isolate the frozen-run filter from Hampel and short-gap interp.
        """
        n = 60
        ramp = list(np.arange(n) * 0.1 + 100.0)
        ramp[:6] = [100.0] * 6
        df = pl.DataFrame({"bds_tas_kt": ramp, "bds_ias_kt": ramp})
        result = clean_bds_speeds(
            df,
            frozen_min_run_len_tas=6,
            frozen_min_run_len_ias=20,
            n_passes=0,
            interp_max_gap=0,
        )

        tas_clean = result["bds_tas_kt_clean"].to_numpy()
        ias_clean = result["bds_ias_kt_clean"].to_numpy()
        # tas: first 6 NaN (run length 6 >= threshold 6)
        assert all(math.isnan(v) for v in tas_clean[:6])
        # ias: first 6 kept (run length 6 < threshold 20)
        for v in ias_clean[:6]:
            assert v == pytest.approx(100.0)
