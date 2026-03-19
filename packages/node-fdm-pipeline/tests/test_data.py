"""Tests for data pipeline commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from node_fdm_pipeline.commands.data import (
    _require_traffic,
    aircraft_list,
    derive,
    download,
    identify,
)


class TestRequireTraffic:
    """Tests for the traffic import guard."""

    def test_require_traffic_missing(self) -> None:
        """Raises SystemExit when traffic is not installed."""
        with (
            patch.dict("sys.modules", {"traffic": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'traffic'")),
            pytest.raises(SystemExit, match="1"),
        ):
            _require_traffic()


class TestDownloadCommand:
    """Tests for the ``download`` command."""

    def test_download_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without writing files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "aircraft_db.csv").write_text(
            "icao24,registration,typecode,age,airline\nabc123,F-WXYZ,A320,5,AFR\n"
        )
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  download_dir: "download"

typecodes:
  - A320
"""
        )

        download(
            config=config,
            start_date="2025-01-01",
            end_date="2025-01-02",
            dry_run=True,
        )

        # No download dir created in dry-run
        download_dir = data_dir / "download"
        if download_dir.exists():
            assert len(list(download_dir.iterdir())) == 0

    def test_download_missing_aircraft_db(self, tmp_path: Path) -> None:
        """Raises SystemExit when aircraft_db.csv is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  download_dir: "download"

typecodes:
  - A320
"""
        )

        with pytest.raises(SystemExit):
            download(
                config=config,
                start_date="2025-01-01",
                end_date="2025-01-02",
                dry_run=False,
            )


class TestAircraftListCommand:
    """Tests for the ``aircraft-list`` command."""

    def test_aircraft_list_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without querying OpenSky."""
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{tmp_path / "data"}"

typecodes:
  - A320
"""
        )

        # dry_run returns before _require_traffic() is called
        aircraft_list(config=config, dry_run=True)


class TestCLINewCommands:
    """CLI smoke tests for new data commands."""

    def test_aircraft_list_help(self) -> None:
        """fdm aircraft-list --help shows expected flags."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "aircraft-list", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--config" in output
        assert "--dry-run" in output

    def test_download_help(self) -> None:
        """fdm download --help shows expected flags."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "download", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--start-date" in output
        assert "--end-date" in output
        assert "--dry-run" in output


# ---------------------------------------------------------------------------
# Download → Delta Table tests (v3 étape 0)
# ---------------------------------------------------------------------------


class TestDownloadDelta:
    """Tests for download → Delta Table (v3 pipeline étape 0)."""

    @staticmethod
    def _make_raw_df(n: int = 10) -> pl.DataFrame:
        """Synthetic OpenSky data with raw + BDS columns (pre-rename)."""
        from datetime import datetime as dt

        return pl.DataFrame(
            {
                "timestamp": [dt(2025, 1, 1, 12, 0, i) for i in range(n)],
                "icao24": ["abc123"] * n,
                "callsign": ["TEST01"] * n,
                "latitude": [48.0 + i * 0.001 for i in range(n)],
                "longitude": [2.0 + i * 0.001 for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                "track": [90.0] * n,
                "vertical_rate": [100.0] * n,
                "selected_mcp": [35000.0] * n,
                "selected_fms": [35000.0] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "Mach": [0.78] * n,
                "heading": [88.0] * n,
            }
        )

    def test_download_creates_delta(self, tmp_path: Path) -> None:
        """Delta Table created with raw_* + bds_* columns."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        df = _rename_to_v3(self._make_raw_df(), batch_date="20250101")
        table_path = tmp_path / "flights.delta"
        write_columns(df, table_path)

        result = pl.read_delta(str(table_path))
        raw_cols = {c for c in result.columns if c.startswith("raw_")}
        bds_cols = {c for c in result.columns if c.startswith("bds_")}
        assert raw_cols == {
            "raw_timestamp",
            "raw_icao24",
            "raw_callsign",
            "raw_lat_deg",
            "raw_lon_deg",
            "raw_alt_ft",
            "raw_gs_kt",
            "raw_track_deg",
            "raw_vz_ftmin",
        }
        assert bds_cols == {
            "bds_mcp_sel_alt_ft",
            "bds_fms_sel_alt_ft",
            "bds_ias_kt",
            "bds_tas_kt",
            "bds_mach",
            "bds_hdg_deg",
        }
        assert "meta_batch_date" in result.columns

    def test_download_partition_key(self, tmp_path: Path) -> None:
        """2 dates → 2 partitions meta_batch_date."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        table_path = tmp_path / "flights.delta"
        df1 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250101")
        df2 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250102")
        combined = pl.concat([df1, df2])
        write_columns(combined, table_path)

        result = pl.read_delta(str(table_path))
        dates = result["meta_batch_date"].unique().sort().to_list()
        assert dates == ["20250101", "20250102"]

    def test_download_bds_preserved(self, tmp_path: Path) -> None:
        """BDS TAS and Mach preserved with correct column names."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        df = _rename_to_v3(self._make_raw_df(), batch_date="20250101")
        table_path = tmp_path / "flights.delta"
        write_columns(df, table_path)

        result = pl.read_delta(str(table_path))
        assert "bds_tas_kt" in result.columns
        assert "bds_mach" in result.columns
        assert result["bds_tas_kt"].to_list() == [450.0] * 10
        assert result["bds_mach"].to_list() == [0.78] * 10

    def test_download_idempotent(self, tmp_path: Path) -> None:
        """Download 2x same date -> partition overwritten, no duplicates."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        table_path = tmp_path / "flights.delta"

        # First write
        df1 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250101")
        write_columns(df1, table_path)

        # Second write (same date, same schema)
        df2 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250101")
        write_columns(df2, table_path)

        result = pl.read_delta(str(table_path))
        assert len(result) == 5  # overwritten, not appended


# ---------------------------------------------------------------------------
# identify tests (v3 étape 1)
# ---------------------------------------------------------------------------


def _make_delta_table(
    tmp_path: Path,
    *,
    icao24s: list[str],
    callsigns: list[str | None],
    timestamps_s: list[list[int]],
    batch_date: str = "20250101",
) -> Path:
    """Create a synthetic Delta Table with raw_* columns for identify tests.

    Each entry in icao24s/callsigns/timestamps_s defines one flight's points.
    """
    from datetime import UTC, timedelta
    from datetime import datetime as dt

    rows: dict[str, list[object]] = {
        "raw_timestamp": [],
        "raw_icao24": [],
        "raw_callsign": [],
        "raw_lat_deg": [],
        "raw_lon_deg": [],
        "raw_alt_ft": [],
        "raw_gs_kt": [],
        "raw_track_deg": [],
        "raw_vz_ftmin": [],
        "meta_batch_date": [],
    }
    base = dt(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    for icao24, callsign, ts_offsets in zip(icao24s, callsigns, timestamps_s, strict=False):
        for offset in ts_offsets:
            rows["raw_timestamp"].append(base + timedelta(seconds=offset))
            rows["raw_icao24"].append(icao24)
            rows["raw_callsign"].append(callsign)
            rows["raw_lat_deg"].append(48.0)
            rows["raw_lon_deg"].append(2.0)
            rows["raw_alt_ft"].append(35000.0)
            rows["raw_gs_kt"].append(440.0)
            rows["raw_track_deg"].append(90.0)
            rows["raw_vz_ftmin"].append(100.0)
            rows["meta_batch_date"].append(batch_date)

    df = pl.DataFrame(rows).cast({"raw_callsign": pl.Utf8})
    table_path = tmp_path / "flights.delta"
    from node_fdm_data.delta import write_columns

    write_columns(df, table_path)
    return table_path


def _make_flightlist(
    tmp_path: Path,
    *,
    entries: list[dict[str, str]],
    batch_date: str = "20250101",
) -> Path:
    """Create a synthetic flightlist parquet.

    Each entry: {icao24, callsign, departure, arrival, typecode, firstseen, lastseen}.
    """
    from datetime import UTC
    from datetime import datetime as dt

    rows: dict[str, list[object]] = {
        "icao24": [],
        "callsign": [],
        "departure": [],
        "arrival": [],
        "typecode": [],
        "firstseen": [],
        "lastseen": [],
    }
    for e in entries:
        rows["icao24"].append(e["icao24"])
        rows["callsign"].append(e.get("callsign", ""))
        rows["departure"].append(e.get("departure"))
        rows["arrival"].append(e.get("arrival"))
        rows["typecode"].append(e.get("typecode"))
        rows["firstseen"].append(
            dt(2025, 1, 1, 11, 0, 0, tzinfo=UTC),
        )
        rows["lastseen"].append(
            dt(2025, 1, 1, 13, 0, 0, tzinfo=UTC),
        )

    df = pl.DataFrame(rows)
    download_dir = tmp_path / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    fl_path = download_dir / f"flightlist_{batch_date}.parquet"
    df.write_parquet(fl_path)
    return download_dir


class TestIdentify:
    """Tests for the ``identify`` command (v3 étape 1)."""

    @staticmethod
    def _make_config(tmp_path: Path, *, data_dir: Path) -> Path:
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  download_dir: "download"

typecodes:
  - A320
"""
        )
        return config

    def test_identify_flight_id_format(self, tmp_path: Path) -> None:
        """meta_flight_id follows {icao24}_{callsign}_s{idx} format."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Two icao24s, each with a gap → 2 segments each = 4 total segments
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123", "abc123", "def456"],
            callsigns=["TEST01", "TEST01", "FLY02"],
            timestamps_s=[
                list(range(0, 20)),  # abc123 seg 1: 0-19s
                list(range(60, 80)),  # abc123 seg 2: 60-79s (gap > 30s)
                list(range(0, 15)),  # def456 seg 1: 0-14s
            ],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert "meta_flight_id" in result.columns
        assert "meta_original_flight_id" in result.columns

        flight_ids = result["meta_flight_id"].unique().sort().to_list()
        # abc123_TEST01 → 2 segments (_s0, _s1), def456_FLY02 → 1 segment (_s0)
        assert "abc123_TEST01_s0" in flight_ids
        assert "abc123_TEST01_s1" in flight_ids
        assert "def456_FLY02_s0" in flight_ids

        # original_flight_id has no segment suffix
        orig_ids = result["meta_original_flight_id"].unique().sort().to_list()
        assert orig_ids == ["abc123_TEST01", "def456_FLY02"]

    def test_identify_keeps_short_segments(self, tmp_path: Path) -> None:
        """Segments with few points are NOT filtered — they get a flight_id."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 10 points, then gap, then 5 points (short segment)
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123", "abc123"],
            callsigns=["TEST01", "TEST01"],
            timestamps_s=[
                list(range(0, 10)),
                list(range(60, 65)),  # only 5 points — short segment
            ],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        # All 15 rows kept
        assert len(result) == 15
        # Both segments have a flight_id
        ids = result["meta_flight_id"].unique().to_list()
        assert len(ids) == 2

    def test_identify_flightlist_join(self, tmp_path: Path) -> None:
        """Flightlist metadata joined → meta_departure, meta_arrival, meta_aircraft_type."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(
            data_dir,
            entries=[
                {
                    "icao24": "abc123",
                    "callsign": "TEST01",
                    "departure": "LFPG",
                    "arrival": "EGLL",
                    "typecode": "A320",
                },
            ],
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert result["meta_departure"][0] == "LFPG"
        assert result["meta_arrival"][0] == "EGLL"
        assert result["meta_aircraft_type"][0] == "A320"

    def test_identify_no_flightlist(self, tmp_path: Path) -> None:
        """icao24 absent from flightlist → meta_departure/arrival/aircraft_type = null."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(data_dir, entries=[])  # empty flightlist
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert result["meta_departure"].null_count() == len(result)
        assert result["meta_arrival"].null_count() == len(result)
        assert result["meta_aircraft_type"].null_count() == len(result)

    def test_identify_gap_below_threshold(self, tmp_path: Path) -> None:
        """Gap of 29s (< 30s threshold) → no split, same segment."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Points at 0-9s, then 38s (29s gap from t=9)
        timestamps = [*range(0, 10), 38]
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[timestamps],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        ids = result["meta_flight_id"].unique().to_list()
        assert len(ids) == 1  # single segment

    def test_identify_callsign_null(self, tmp_path: Path) -> None:
        """Null callsign → meta_flight_id uses NOCALL placeholder."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=[None],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        fid = result["meta_flight_id"][0]
        assert "NOCALL" in fid
        assert fid == "abc123_NOCALL_s0"

    def test_identify_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without modifying the Delta Table."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=True)

        result = pl.read_delta(str(table_path))
        assert "meta_flight_id" not in result.columns


# ---------------------------------------------------------------------------
# derive tests (v3 étape 4)
# ---------------------------------------------------------------------------


def _make_derive_delta_table(tmp_path: Path, *, n_flights: int = 1) -> Path:
    """Create a Delta Table with all columns needed for the derive step."""
    import numpy as np

    rows_per_flight = 10
    all_rows: dict[str, list[object]] = {
        "raw_timestamp": [],
        "raw_icao24": [],
        "raw_callsign": [],
        "raw_lat_deg": [],
        "raw_lon_deg": [],
        "raw_alt_ft": [],
        "raw_gs_kt": [],
        "raw_track_deg": [],
        "raw_vz_ftmin": [],
        "bds_mcp_sel_alt_ft": [],
        "era_tas_kt": [],
        "meta_flight_id": [],
        "meta_departure": [],
        "meta_arrival": [],
        "meta_batch_date": [],
    }
    from datetime import UTC, timedelta
    from datetime import datetime as dt

    base = dt(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    for fi in range(n_flights):
        lats = np.linspace(48.0, 48.1, rows_per_flight)
        lons = np.linspace(2.0, 2.1, rows_per_flight)
        for i in range(rows_per_flight):
            all_rows["raw_timestamp"].append(base + timedelta(seconds=fi * 100 + i))
            all_rows["raw_icao24"].append(f"abc{fi:03d}")
            all_rows["raw_callsign"].append(f"TST{fi:02d}")
            all_rows["raw_lat_deg"].append(float(lats[i]))
            all_rows["raw_lon_deg"].append(float(lons[i]))
            all_rows["raw_alt_ft"].append(35000.0)
            all_rows["raw_gs_kt"].append(440.0)
            all_rows["raw_track_deg"].append(90.0)
            all_rows["raw_vz_ftmin"].append(500.0)
            all_rows["bds_mcp_sel_alt_ft"].append(36000.0)
            all_rows["era_tas_kt"].append(450.0)
            all_rows["meta_flight_id"].append(f"abc{fi:03d}_TST{fi:02d}_s0")
            all_rows["meta_departure"].append("LFPG")
            all_rows["meta_arrival"].append("EGLL")
            all_rows["meta_batch_date"].append("20250101")

    df = pl.DataFrame(all_rows)
    table_path = tmp_path / "flights.delta"
    from node_fdm_data.delta import write_columns

    write_columns(df, table_path)
    return table_path


class TestDeriveCommand:
    """Tests for the ``derive`` command (v3 étape 4)."""

    @staticmethod
    def _make_config(tmp_path: Path, *, data_dir: Path) -> Path:
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )
        return config

    def test_derive_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without modifying the Delta Table."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        derive(config=config, dry_run=True)

        result = pl.read_delta(str(table_path))
        assert "fdm_gamma_rad" not in result.columns

    def test_derive_adds_columns(self, tmp_path: Path) -> None:
        """derive adds fdm_gamma_rad, fdm_long_wind_kt, fdm_alt_diff_ft, fdm_distance_cum_m."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        derive(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        for col in (
            "fdm_gamma_rad",
            "fdm_long_wind_kt",
            "fdm_alt_diff_ft",
            "fdm_distance_cum_m",
            "fdm_adep_dist_nm",
            "fdm_ades_dist_nm",
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_derive_per_flight_distance(self, tmp_path: Path) -> None:
        """fdm_distance_cum_m resets to 0 for each meta_flight_id."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir, n_flights=2)
        config = self._make_config(tmp_path, data_dir=data_dir)

        derive(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        for fid in result["meta_flight_id"].unique().to_list():
            flight = result.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
            assert flight["fdm_distance_cum_m"][0] == 0.0

    def test_derive_idempotent(self, tmp_path: Path) -> None:
        """Running derive twice produces the same result (idempotence)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        derive(config=config, dry_run=False)
        first = pl.read_delta(str(table_path))

        derive(config=config, dry_run=False)
        second = pl.read_delta(str(table_path))

        for col in ("fdm_gamma_rad", "fdm_long_wind_kt", "fdm_alt_diff_ft", "fdm_distance_cum_m"):
            assert first[col].to_list() == second[col].to_list(), f"{col} changed on re-run"
