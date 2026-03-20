"""Tests for node_fdm_data.preprocessing.resample — gap-aware resampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from node_fdm_data.preprocessing.resample import (
    detect_subsegments,
    interpolate_group_by_subsegments,
    preprocess_flights,
    resample_flight,
)


@dataclass
class _FlightSpec:
    """Optional overrides for :func:`_make_flight`."""

    ts_start: datetime = field(default_factory=lambda: datetime(2025, 1, 1))
    ts_step_s: int = 1
    lat_start: float = 48.0
    lat_step: float = 0.001
    with_bds: bool = False


def _make_flight(
    flight_id: str,
    n: int,
    spec: _FlightSpec | None = None,
) -> pl.DataFrame:
    """Build a synthetic flight DataFrame for resample tests."""
    s = spec or _FlightSpec()

    timestamps = [s.ts_start + timedelta(seconds=i * s.ts_step_s) for i in range(n)]
    lats: list[float | None] = [s.lat_start + i * s.lat_step for i in range(n)]
    lons: list[float | None] = [2.0 + i * s.lat_step * 0.5 for i in range(n)]

    data: dict[str, list[Any]] = {
        "meta_flight_id": [flight_id] * n,
        "raw_timestamp": timestamps,
        "raw_lat_deg": lats,
        "raw_lon_deg": lons,
        "raw_alt_ft": [35000.0] * n,
        "raw_gs_kt": [450.0] * n,
        "raw_track_deg": [90.0] * n,
        "raw_vz_ftmin": [0.0] * n,
    }

    if s.with_bds:
        data["bds_mach"] = [0.78] * n
        data["bds_tas_kt"] = [450.0] * n
        data["bds_ias_kt"] = [280.0] * n
        data["bds_hdg_deg"] = [90.0] * n
        data["bds_mcp_sel_alt_ft"] = [35000.0] * n
        data["bds_fms_sel_alt_ft"] = [35000.0] * n

    return pl.DataFrame(data)


class TestDetectSubsegmentsSingleRun:
    """All non-null data → single sub-segment covering all rows."""

    def test_single_run(self) -> None:
        """Continuous non-null lat/lon → one sub-segment (ID 0)."""
        df = _make_flight("f1", 20)
        seg_ids = detect_subsegments(df, ["raw_lat_deg", "raw_lon_deg"], max_gap_s=30)
        unique = set(seg_ids.to_list())
        assert unique == {0}


class TestDetectSubsegmentsGapSplit:
    """Gap > max_gap_s splits into two sub-segments."""

    def test_gap_split(self) -> None:
        """60s gap with max_gap=30 → two sub-segments."""
        ts_start = datetime(2025, 1, 1)
        part1 = _make_flight("f1", 10, _FlightSpec(ts_start=ts_start))
        part2 = _make_flight(
            "f1",
            10,
            _FlightSpec(ts_start=ts_start + timedelta(seconds=70), lat_start=49.0),
        )
        df = pl.concat([part1, part2]).sort("raw_timestamp")
        seg_ids = detect_subsegments(df, ["raw_lat_deg", "raw_lon_deg"], max_gap_s=30)
        unique_segs = set(seg_ids.to_list()) - {-1}
        assert len(unique_segs) == 2


class TestDetectSubsegmentsIsolatedPoints:
    """Single isolated point between gaps → excluded (seg_id = -1)."""

    def test_isolated_point_excluded(self) -> None:
        """1 point between two 60s gaps → segment too short, set to -1."""
        ts_start = datetime(2025, 1, 1)
        part1 = _make_flight("f1", 5, _FlightSpec(ts_start=ts_start))
        isolated = pl.DataFrame(
            {
                "meta_flight_id": ["f1"],
                "raw_timestamp": [ts_start + timedelta(seconds=65)],
                "raw_lat_deg": [48.5],
                "raw_lon_deg": [2.5],
                "raw_alt_ft": [35000.0],
                "raw_gs_kt": [450.0],
                "raw_track_deg": [90.0],
                "raw_vz_ftmin": [0.0],
            }
        )
        part2 = _make_flight(
            "f1",
            5,
            _FlightSpec(ts_start=ts_start + timedelta(seconds=130), lat_start=49.0),
        )
        df = pl.concat([part1, isolated, part2], how="diagonal_relaxed").sort("raw_timestamp")
        seg_ids = detect_subsegments(df, ["raw_lat_deg", "raw_lon_deg"], max_gap_s=30)
        # The isolated point at index 5 should be -1
        assert seg_ids[5] == -1
        # The other two clusters should be valid segments
        valid_segs = set(seg_ids.to_list()) - {-1}
        assert len(valid_segs) == 2


class TestInterpolateWithinSubsegment:
    """Interpolation fills nulls within a single sub-segment."""

    def test_interpolate_fills_gap_on_grid(self) -> None:
        """Grid point between two data points → interpolated linearly."""
        ts_start = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * 3,
                "raw_timestamp": [
                    ts_start,
                    ts_start + timedelta(seconds=4),
                    ts_start + timedelta(seconds=8),
                ],
                "raw_lat_deg": [48.0, None, 48.002],
                "raw_lon_deg": [2.0, None, 2.002],
            }
        )
        # Segment 0 = rows 0 and 2; row 1 has null data → seg_id -1
        seg_ids = pl.Series("seg_id", [0, -1, 0], dtype=pl.Int32)
        grid_ts = pl.Series(
            "raw_timestamp",
            [
                ts_start,
                ts_start + timedelta(seconds=4),
                ts_start + timedelta(seconds=8),
            ],
        )
        col_values, gap_flag = interpolate_group_by_subsegments(
            df,
            grid_ts,
            ["raw_lat_deg", "raw_lon_deg"],
            seg_ids,
        )
        # Middle point should be interpolated to midpoint
        assert col_values["raw_lat_deg"][1] is not None
        assert abs(float(col_values["raw_lat_deg"][1]) - 48.001) < 0.0001  # type: ignore[arg-type]
        assert abs(float(col_values["raw_lon_deg"][1]) - 2.001) < 0.0001  # type: ignore[arg-type]
        # All grid points covered → no gap
        assert gap_flag.to_list() == [False, False, False]


class TestNoInterpolationAcrossGap:
    """Values between sub-segments must remain null with gap flag."""

    def test_null_and_flag_in_gap(self) -> None:
        """Two sub-segments with 120s gap → null + pre_gap_position in between."""
        ts_start = datetime(2025, 1, 1)
        part1 = _make_flight("f1", 30, _FlightSpec(ts_start=ts_start))
        part2 = _make_flight(
            "f1",
            30,
            _FlightSpec(ts_start=ts_start + timedelta(seconds=150), lat_start=49.0),
        )
        df = pl.concat([part1, part2]).sort("raw_timestamp")

        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)

        gap_rows = result.filter(pl.col("pre_gap_position"))
        assert len(gap_rows) > 0
        assert gap_rows["raw_lat_deg"].is_null().all()


class TestResampleFlightRegularGrid:
    """resample_flight produces a regular 4s grid."""

    def test_regular_grid(self) -> None:
        """Resampled flight has constant 4s timestamp diff."""
        df = _make_flight("f1", 600)
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        diffs = result["raw_timestamp"].diff().dt.total_seconds().drop_nulls()
        assert (diffs == 4.0).all()


class TestPreprocessDropsShortFlights:
    """preprocess_flights drops flights shorter than min_duration_s."""

    def test_short_flight_dropped(self) -> None:
        """Flight of 2 min dropped, flight of 10 min kept."""
        short = _make_flight("short", 120)  # 119s
        long = _make_flight("long", 600, _FlightSpec(lat_start=49.0))  # 599s
        df = pl.concat([short, long], how="diagonal_relaxed")

        result = preprocess_flights(df, min_duration_s=240, smooth=False)
        flights = result["meta_flight_id"].unique().to_list()
        assert "long" in flights
        assert "short" not in flights


class TestGapFlagsCorrect:
    """pre_gap_* flags are correctly positioned per group."""

    def test_gap_flags_independent(self) -> None:
        """Position sparse, altitude dense → different gap flag patterns."""
        ts_start = datetime(2025, 1, 1)
        n = 600
        timestamps = [ts_start + timedelta(seconds=i) for i in range(n)]

        # Altitude: all present
        alts: list[float | None] = [35000.0] * n
        gs: list[float | None] = [450.0] * n
        tracks: list[float | None] = [90.0] * n
        vzs: list[float | None] = [0.0] * n

        # Position: first 30 and last 30 points only
        lats: list[float | None] = [None] * n
        lons: list[float | None] = [None] * n
        for i in range(30):
            lats[i] = 48.0 + i * 0.001
            lons[i] = 2.0 + i * 0.001
        for i in range(n - 30, n):
            lats[i] = 49.0 + (i - (n - 30)) * 0.001
            lons[i] = 3.0 + (i - (n - 30)) * 0.001

        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * n,
                "raw_timestamp": timestamps,
                "raw_lat_deg": lats,
                "raw_lon_deg": lons,
                "raw_alt_ft": alts,
                "raw_gs_kt": gs,
                "raw_track_deg": tracks,
                "raw_vz_ftmin": vzs,
            }
        )

        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)

        pos_gap_pct = result["pre_gap_position"].sum() / len(result)
        alt_gap_pct = result["pre_gap_altitude"].sum() / len(result)

        # Position has big gap in the middle → mostly gap
        assert pos_gap_pct > 0.5
        # Altitude is dense → almost no gap
        assert alt_gap_pct < 0.05


class TestPreprocessIdempotent:
    """Running preprocess twice gives the same result."""

    def test_idempotent(self) -> None:
        """Second preprocess on already-preprocessed data → same result."""
        df = _make_flight("f1", 600)
        result1 = preprocess_flights(df, min_duration_s=240, smooth=False)
        result2 = preprocess_flights(result1, min_duration_s=240, smooth=False)
        assert result1.shape == result2.shape
        for c in result1.columns:
            if result1[c].dtype.is_float():
                s1 = result1[c].fill_null(0.0)
                s2 = result2[c].fill_null(0.0)
                diff = (s1 - s2).abs().max()
                assert diff is None or float(diff) < 1e-6  # type: ignore[arg-type]


class TestDetectSubsegmentsNoCols:
    """Reference columns missing from DataFrame → all -1."""

    def test_missing_ref_cols(self) -> None:
        """If ref_cols not in DataFrame → all seg_ids are -1."""
        df = _make_flight("f1", 20)
        seg_ids = detect_subsegments(df, ["nonexistent_col"], max_gap_s=30)
        assert set(seg_ids.to_list()) == {-1}


class TestDetectSubsegmentsAllNull:
    """All reference column values are null → all -1."""

    def test_all_null_ref(self) -> None:
        ts_start = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * 10,
                "raw_timestamp": [ts_start + timedelta(seconds=i) for i in range(10)],
                "raw_lat_deg": [None] * 10,
                "raw_lon_deg": [None] * 10,
            }
        )
        seg_ids = detect_subsegments(df, ["raw_lat_deg", "raw_lon_deg"], max_gap_s=30)
        assert set(seg_ids.to_list()) == {-1}


class TestInterpolateNoSegments:
    """No valid sub-segments → all null, all gap."""

    def test_all_gap_when_no_segments(self) -> None:
        ts_start = datetime(2025, 1, 1)
        df = pl.DataFrame(
            {
                "raw_timestamp": [ts_start + timedelta(seconds=i) for i in range(10)],
                "raw_lat_deg": [None] * 10,
            }
        )
        grid_ts = pl.Series(
            "raw_timestamp",
            [ts_start + timedelta(seconds=i * 4) for i in range(3)],
        )
        seg_ids = pl.Series("seg_id", [-1] * 10, dtype=pl.Int32)
        col_values, gap_flag = interpolate_group_by_subsegments(
            df,
            grid_ts,
            ["raw_lat_deg"],
            seg_ids,
        )
        assert all(v is None for v in col_values["raw_lat_deg"])
        assert gap_flag.all()


class TestResampleWithBDSColumns:
    """BDS columns are interpolated and get their own gap flag."""

    def test_bds_group_interpolated(self) -> None:
        """Flight with BDS data → bds columns present, pre_gap_bds flag exists."""
        df = _make_flight("f1", 600, _FlightSpec(with_bds=True))
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert "pre_gap_bds" in result.columns
        assert "bds_mach" in result.columns
        # BDS data is dense → almost no gap
        assert result["pre_gap_bds"].sum() / len(result) < 0.05

    def test_bds_sparse_has_gaps(self) -> None:
        """BDS data with big gap → pre_gap_bds True in the middle."""
        ts_start = datetime(2025, 1, 1)
        n = 600
        timestamps = [ts_start + timedelta(seconds=i) for i in range(n)]
        bds_mach: list[float | None] = [None] * n
        # Only first 30 and last 30 have BDS
        for i in range(30):
            bds_mach[i] = 0.78
        for i in range(n - 30, n):
            bds_mach[i] = 0.79

        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * n,
                "raw_timestamp": timestamps,
                "raw_alt_ft": [35000.0] * n,
                "raw_gs_kt": [450.0] * n,
                "raw_track_deg": [90.0] * n,
                "raw_vz_ftmin": [0.0] * n,
                "bds_mach": bds_mach,
                "bds_tas_kt": bds_mach,  # reuse pattern
                "bds_ias_kt": [None] * n,
                "bds_hdg_deg": [None] * n,
                "bds_mcp_sel_alt_ft": [None] * n,
                "bds_fms_sel_alt_ft": [None] * n,
            }
        )
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        # Big gap in the middle → pre_gap_bds should have True values
        assert result["pre_gap_bds"].sum() > 0


class TestResampleMetaColumnsCarried:
    """Meta columns are carried over to resampled result."""

    def test_meta_preserved(self) -> None:
        df = _make_flight("f1", 600)
        df = df.with_columns(
            pl.lit("LFPG").alias("meta_departure"),
            pl.lit("EGLL").alias("meta_arrival"),
        )
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert "meta_departure" in result.columns
        assert result["meta_departure"][0] == "LFPG"
        assert result["meta_arrival"][0] == "EGLL"
        # All rows have the same meta value
        assert result["meta_departure"].n_unique() == 1


class TestPreprocessEmptyResult:
    """All flights too short → empty DataFrame returned."""

    def test_all_flights_too_short(self) -> None:
        short1 = _make_flight("s1", 60)  # 59s
        short2 = _make_flight("s2", 100, _FlightSpec(lat_start=49.0))  # 99s
        df = pl.concat([short1, short2], how="diagonal_relaxed")
        result = preprocess_flights(df, min_duration_s=240, smooth=False)
        assert len(result) == 0


class TestPreprocessMultipleFlights:
    """Multiple flights are independently resampled."""

    def test_two_flights(self) -> None:
        f1 = _make_flight("f1", 600)
        f2 = _make_flight("f2", 800, _FlightSpec(lat_start=49.0))
        df = pl.concat([f1, f2], how="diagonal_relaxed")
        result = preprocess_flights(df, min_duration_s=240, smooth=False)
        flights = result["meta_flight_id"].unique().sort().to_list()
        assert flights == ["f1", "f2"]
        # Each flight should have regular 4s grid
        for fid in flights:
            sub = result.filter(pl.col("meta_flight_id") == fid)
            diffs = sub["raw_timestamp"].diff().dt.total_seconds().drop_nulls()
            assert (diffs == 4.0).all()


class TestEdgeCases:
    """Edge cases from the test specification."""

    def test_all_position_null(self) -> None:
        """Flight with all lat/lon null → kept, pre_gap_position all True."""
        ts_start = datetime(2025, 1, 1)
        n = 600
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * n,
                "raw_timestamp": [ts_start + timedelta(seconds=i) for i in range(n)],
                "raw_lat_deg": [None] * n,
                "raw_lon_deg": [None] * n,
                "raw_alt_ft": [35000.0] * n,
                "raw_gs_kt": [450.0] * n,
                "raw_track_deg": [90.0] * n,
                "raw_vz_ftmin": [0.0] * n,
            }
        )
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert result["pre_gap_position"].all()
        # Altitude should be mostly non-gap
        assert not result["pre_gap_altitude"].all()

    def test_already_on_grid(self) -> None:
        """Data already at 4s intervals → same grid, same count."""
        df = _make_flight("f1", 150, _FlightSpec(ts_step_s=4))  # 596s at 4s
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert len(result) == 150
        diffs = result["raw_timestamp"].diff().dt.total_seconds().drop_nulls()
        assert (diffs == 4.0).all()

    def test_single_point_segment_not_interpolated(self) -> None:
        """Single isolated non-null point → not a valid sub-segment."""
        ts_start = datetime(2025, 1, 1)
        n = 600
        lats: list[float | None] = [None] * n
        lons: list[float | None] = [None] * n
        # Only one point has position
        lats[300] = 48.0
        lons[300] = 2.0

        df = pl.DataFrame(
            {
                "meta_flight_id": ["f1"] * n,
                "raw_timestamp": [ts_start + timedelta(seconds=i) for i in range(n)],
                "raw_lat_deg": lats,
                "raw_lon_deg": lons,
                "raw_alt_ft": [35000.0] * n,
                "raw_gs_kt": [450.0] * n,
                "raw_track_deg": [90.0] * n,
                "raw_vz_ftmin": [0.0] * n,
            }
        )
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        # Single point excluded → all position is gap
        assert result["pre_gap_position"].all()


class TestResamplePreservesStringIdentifiers:
    """raw_icao24 and raw_callsign must survive resampling."""

    def test_resample_preserves_icao24(self) -> None:
        """raw_icao24 is carried over with correct value, no nulls."""
        n = 100
        df = _make_flight("f1", n)
        df = df.with_columns(pl.lit("3c6634").alias("raw_icao24"))
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert "raw_icao24" in result.columns
        assert result["raw_icao24"].null_count() == 0
        assert result["raw_icao24"][0] == "3c6634"
        assert result["raw_icao24"].n_unique() == 1

    def test_resample_preserves_callsign(self) -> None:
        """raw_callsign is carried over with correct value, no nulls."""
        df = _make_flight("f1", 100)
        df = df.with_columns(pl.lit("DLH123").alias("raw_callsign"))
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert "raw_callsign" in result.columns
        assert result["raw_callsign"].null_count() == 0
        assert result["raw_callsign"][0] == "DLH123"
        assert result["raw_callsign"].n_unique() == 1

    def test_resample_absent_icao24(self) -> None:
        """No raw_icao24 in source → no error, column not in output."""
        df = _make_flight("f1", 100)
        assert "raw_icao24" not in df.columns
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert "raw_icao24" not in result.columns

    def test_resample_null_icao24(self) -> None:
        """Null raw_icao24 in source → null preserved in output."""
        df = _make_flight("f1", 100)
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("raw_icao24"))
        result = resample_flight(df, rate_s=4, max_gap_s=30, smooth=False)
        assert "raw_icao24" in result.columns
        assert result["raw_icao24"].null_count() == len(result)
