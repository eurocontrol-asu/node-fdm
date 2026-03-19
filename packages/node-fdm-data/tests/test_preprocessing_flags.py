"""Tests for node_fdm_data.preprocessing.flags — validity flags (étape 2)."""

from __future__ import annotations

import polars as pl

from node_fdm_data.preprocessing.flags import compute_flags


def _make_flight(
    flight_id: str,
    n: int,
    *,
    gs_kt: float = 200.0,
    lat_start: float = 48.0,
    lat_step: float = 0.01,
) -> pl.DataFrame:
    """Helper: build a minimal flight DataFrame for flag tests."""
    lats = [lat_start + i * lat_step for i in range(n)]
    return pl.DataFrame(
        {
            "meta_flight_id": [flight_id] * n,
            "raw_timestamp": list(range(n)),
            "raw_gs_kt": [gs_kt] * n,
            "raw_lat_deg": lats,
            "raw_lon_deg": [2.0] * n,
        }
    )


class TestFlagMinPoints:
    """fdm_flag_min_points: segment has >= N points."""

    def test_flag_min_points(self) -> None:
        """Segment of 10 points with threshold=40 → flag is false."""
        df = _make_flight("f1", 10)
        result = compute_flags(df, min_points=40)
        assert result["fdm_flag_min_points"].all() is False

    def test_flag_min_points_above_threshold(self) -> None:
        """Segment with enough points → flag is true."""
        df = _make_flight("f1", 50)
        result = compute_flags(df, min_points=40)
        assert result["fdm_flag_min_points"].all() is True

    def test_flag_min_points_per_flight(self) -> None:
        """Each flight is evaluated independently."""
        short = _make_flight("f_short", 5)
        long = _make_flight("f_long", 50, lat_start=49.0)
        df = pl.concat([short, long])
        result = compute_flags(df, min_points=40)
        short_flags = result.filter(pl.col("meta_flight_id") == "f_short")
        long_flags = result.filter(pl.col("meta_flight_id") == "f_long")
        assert short_flags["fdm_flag_min_points"].all() is False
        assert long_flags["fdm_flag_min_points"].all() is True


class TestFlagMinSpeed:
    """fdm_flag_min_speed: raw_gs_kt > threshold."""

    def test_flag_min_speed(self) -> None:
        """Points with gs < 90kt → false, gs > 90kt → true."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * 4,
                "raw_timestamp": [0, 1, 2, 3],
                "raw_gs_kt": [50.0, 89.9, 90.1, 200.0],
                "raw_lat_deg": [48.0, 48.01, 48.02, 48.03],
                "raw_lon_deg": [2.0, 2.0, 2.0, 2.0],
            }
        )
        result = compute_flags(df, min_points=1, min_speed_kt=90.0)
        flags = result.sort("raw_timestamp")["fdm_flag_min_speed"].to_list()
        assert flags == [False, False, True, True]


class TestFlagDistanceOk:
    """fdm_flag_distance_ok: distance-diff in [LOW_THR, UPPER_THR]."""

    def test_flag_distance_ok_below_threshold(self) -> None:
        """Distance-diff ~150m (< LOW_THR=200) → false."""
        # Points very close together → haversine < 200m
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * 3,
                "raw_timestamp": [0, 1, 2],
                "raw_gs_kt": [200.0] * 3,
                "raw_lat_deg": [48.0, 48.001, 48.002],  # ~111m apart
                "raw_lon_deg": [2.0, 2.0, 2.0],
            }
        )
        result = compute_flags(df, min_points=1)
        flags = result.sort("raw_timestamp")["fdm_flag_distance_ok"].to_list()
        # First row = True (no diff), rest = False (< 200m)
        assert flags[0] is True
        assert flags[1] is False
        assert flags[2] is False

    def test_flag_distance_ok_in_range(self) -> None:
        """Distance-diff ~1100m (in [200, 3000]) → true."""
        # ~0.01° lat ≈ 1111m
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * 3,
                "raw_timestamp": [0, 1, 2],
                "raw_gs_kt": [200.0] * 3,
                "raw_lat_deg": [48.0, 48.01, 48.02],
                "raw_lon_deg": [2.0, 2.0, 2.0],
            }
        )
        result = compute_flags(df, min_points=1)
        flags = result.sort("raw_timestamp")["fdm_flag_distance_ok"].to_list()
        assert flags[0] is True  # first row
        assert flags[1] is True  # ~1111m in [200, 3000]
        assert flags[2] is True

    def test_distance_null_first_row(self) -> None:
        """First row of a flight has distance_ok = true (no diff to calculate)."""
        df = _make_flight("f1", 5)
        result = compute_flags(df, min_points=1)
        first_row = result.sort("raw_timestamp").row(0, named=True)
        assert first_row["fdm_flag_distance_ok"] is True


class TestFlagValid:
    """fdm_flag_valid = AND of all flags."""

    def test_flag_valid_and(self) -> None:
        """Mix of flags true/false → fdm_flag_valid = AND correct."""
        # Flight with enough points, good speed, good distance
        df = _make_flight("f1", 50)
        result = compute_flags(df, min_points=40, min_speed_kt=90.0)
        # All flags should be true for airborne points with good distances
        valid = result["fdm_flag_valid"]
        # At least some rows should be valid
        assert valid.sum() > 0
        # Where valid is true, all component flags must also be true
        valid_rows = result.filter(pl.col("fdm_flag_valid"))
        assert valid_rows["fdm_flag_min_points"].all() is True
        assert valid_rows["fdm_flag_min_speed"].all() is True
        assert valid_rows["fdm_flag_distance_ok"].all() is True

    def test_flag_valid_false_when_any_flag_false(self) -> None:
        """If any flag is false, fdm_flag_valid is false."""
        # Short segment → min_points false
        df = _make_flight("f1", 5)
        result = compute_flags(df, min_points=40)
        assert result["fdm_flag_valid"].any() is False


class TestFlagDistancePolarsHaversine:
    """Regression: Polars haversine produces same flags as numpy version."""

    def test_flags_distance_with_polars_haversine(self) -> None:
        """Same flags produced with Polars haversine as with prior numpy impl."""
        # 50 points with ~1111m spacing (0.01° lat) → in [200, 3000] range
        df = _make_flight("f1", 50)
        result = compute_flags(df, min_points=1)
        flags = result.sort("raw_timestamp")

        # First row: distance_ok = True (no diff)
        assert flags["fdm_flag_distance_ok"][0] is True
        # Remaining rows: all ~1111m apart → in [200, 3000] → True
        assert flags["fdm_flag_distance_ok"][1:].all()

    def test_flags_distance_with_null_coords(self) -> None:
        """Null coordinates don't crash the distance flag computation."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * 3,
                "raw_timestamp": [0, 1, 2],
                "raw_gs_kt": [200.0] * 3,
                "raw_lat_deg": [48.0, None, 48.02],
                "raw_lon_deg": [2.0, None, 2.0],
            }
        )
        result = compute_flags(df, min_points=1)
        assert len(result) == 3


class TestEdgeCases:
    """Edge cases from the test specification."""

    def test_single_point_segment(self) -> None:
        """Segment of 1 point: all flags false except maybe min_speed."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"],
                "raw_timestamp": [0],
                "raw_gs_kt": [200.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
            }
        )
        result = compute_flags(df, min_points=40, min_speed_kt=90.0)
        row = result.row(0, named=True)
        assert row["fdm_flag_min_points"] is False
        assert row["fdm_flag_min_speed"] is True  # gs=200 > 90
        assert row["fdm_flag_distance_ok"] is False  # single point
        assert row["fdm_flag_valid"] is False

    def test_no_rows_removed(self) -> None:
        """Flag computation never removes rows (AC8)."""
        df = _make_flight("f1", 20)
        result = compute_flags(df, min_points=40)
        assert len(result) == 20

    def test_crop_columns_are_int64(self) -> None:
        """fdm_flag_crop_start and crop_end are Int64."""
        df = _make_flight("f1", 10)
        result = compute_flags(df, min_points=1)
        assert result["fdm_flag_crop_start"].dtype == pl.Int64
        assert result["fdm_flag_crop_end"].dtype == pl.Int64

    def test_multi_flight_independence(self) -> None:
        """Flags are computed per-flight, not globally."""
        good = _make_flight("good", 50)
        bad = _make_flight("bad", 3, lat_start=49.0)
        df = pl.concat([good, bad])
        result = compute_flags(df, min_points=40)
        good_valid = result.filter(pl.col("meta_flight_id") == "good")["fdm_flag_valid"]
        bad_valid = result.filter(pl.col("meta_flight_id") == "bad")["fdm_flag_valid"]
        assert good_valid.sum() > 0
        assert bad_valid.any() is False
