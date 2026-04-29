"""Tests for data pipeline commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from node_fdm_pipeline.commands.data import (
    _join_flightlist_inline,
    _require_traffic,
    aircraft_list,
    convert,
    derive,
    download,
    flag,
    identify,
    preprocess,
    segments,
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

    def test_download_missing_aircraft_db(self, tmp_path: Path) -> None:
        """Raises SystemExit when aircraft_db.csv is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

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
# _join_flightlist_inline tests
# ---------------------------------------------------------------------------


class TestJoinFlightlistInline:
    """Tests for _join_flightlist_inline (flightlist join during download)."""

    @staticmethod
    def _make_batch_df(n: int = 5) -> pl.DataFrame:
        from datetime import datetime as dt

        return pl.DataFrame(
            {
                "raw_icao24": ["abc123"] * n,
                "raw_callsign": ["TEST01"] * n,
                "raw_lat_deg": [48.0] * n,
                "raw_timestamp": [dt(2025, 1, 1, 12, 0, i) for i in range(n)],
                "meta_batch_date": ["20250101"] * n,
            }
        )

    def test_join_with_flightlist(self) -> None:
        """Flightlist metadata joined onto batch DataFrame."""
        import pandas as pd

        df = self._make_batch_df()
        fl = pd.DataFrame(
            {
                "icao24": ["abc123"],
                "callsign": ["TEST01"],
                "departure": ["LFPG"],
                "arrival": ["EGLL"],
                "typecode": ["A320"],
            }
        )
        result = _join_flightlist_inline(df, fl)
        assert result["meta_departure"][0] == "LFPG"
        assert result["meta_arrival"][0] == "EGLL"
        assert result["meta_aircraft_type"][0] == "A320"

    def test_join_with_none(self) -> None:
        """None flightlist → meta columns are null."""
        df = self._make_batch_df()
        result = _join_flightlist_inline(df, None)
        assert "meta_departure" in result.columns
        assert result["meta_departure"].null_count() == len(result)

    def test_join_no_matching_icao(self) -> None:
        """Flightlist with different icao24 → meta columns are null."""
        import pandas as pd

        df = self._make_batch_df()
        fl = pd.DataFrame(
            {
                "icao24": ["zzz999"],
                "callsign": ["OTHER"],
                "departure": ["KJFK"],
                "arrival": ["KLAX"],
                "typecode": ["B738"],
            }
        )
        result = _join_flightlist_inline(df, fl)
        assert result["meta_departure"].null_count() == len(result)

    def test_join_empty_flightlist(self) -> None:
        """Empty flightlist DataFrame → meta columns are null."""
        import pandas as pd

        df = self._make_batch_df()
        fl = pd.DataFrame(
            {"icao24": [], "callsign": [], "departure": [], "arrival": [], "typecode": []}
        )
        result = _join_flightlist_inline(df, fl)
        assert result["meta_departure"].null_count() == len(result)

    def test_join_flightlist_missing_columns(self) -> None:
        """Flightlist without departure/arrival/typecode → meta columns are null."""
        import pandas as pd

        df = self._make_batch_df()
        fl = pd.DataFrame({"icao24": ["abc123"], "callsign": ["TEST01"]})
        result = _join_flightlist_inline(df, fl)
        assert "meta_departure" in result.columns
        assert result["meta_departure"].null_count() == len(result)


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


def _make_delta_table_with_flightlist(  # noqa: PLR0913
    tmp_path: Path,
    *,
    icao24s: list[str],
    callsigns: list[str | None],
    timestamps_s: list[list[int]],
    meta_departure: str | None = None,
    meta_arrival: str | None = None,
    meta_aircraft_type: str | None = None,
    batch_date: str = "20250101",
) -> Path:
    """Create a Delta Table with raw_* + meta_* columns (as download now produces)."""
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
        "meta_departure": [],
        "meta_arrival": [],
        "meta_aircraft_type": [],
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
            rows["meta_departure"].append(meta_departure)
            rows["meta_arrival"].append(meta_arrival)
            rows["meta_aircraft_type"].append(meta_aircraft_type)

    df = pl.DataFrame(rows).cast(
        {
            "raw_callsign": pl.Utf8,
            "meta_departure": pl.Utf8,
            "meta_arrival": pl.Utf8,
            "meta_aircraft_type": pl.Utf8,
        }
    )
    table_path = tmp_path / "flights.delta"
    from node_fdm_data.delta import write_columns

    write_columns(df, table_path)
    return table_path


class TestIdentify:
    """Tests for the ``identify`` command (v3 étape 1)."""

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

    def test_identify_flight_id_format(self, tmp_path: Path) -> None:
        """meta_flight_id follows {icao24}_{callsign}_s{idx} format."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123", "abc123", "def456"],
            callsigns=["TEST01", "TEST01", "FLY02"],
            timestamps_s=[
                list(range(0, 20)),
                list(range(60, 80)),
                list(range(0, 15)),
            ],
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert "meta_flight_id" in result.columns
        assert "meta_original_flight_id" in result.columns

        flight_ids = result["meta_flight_id"].unique().sort().to_list()
        assert "abc123_TEST01_s0" in flight_ids
        assert "abc123_TEST01_s1" in flight_ids
        assert "def456_FLY02_s0" in flight_ids

        orig_ids = result["meta_original_flight_id"].unique().sort().to_list()
        assert orig_ids == ["abc123_TEST01", "def456_FLY02"]

    def test_identify_keeps_short_segments(self, tmp_path: Path) -> None:
        """Segments with few points are NOT filtered — they get a flight_id."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123", "abc123"],
            callsigns=["TEST01", "TEST01"],
            timestamps_s=[
                list(range(0, 10)),
                list(range(60, 65)),
            ],
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert len(result) == 15
        ids = result["meta_flight_id"].unique().to_list()
        assert len(ids) == 2

    def test_identify_flightlist_metadata_preserved(self, tmp_path: Path) -> None:
        """Flightlist metadata from download preserved through identify."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table_with_flightlist(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
            meta_departure="LFPG",
            meta_arrival="EGLL",
            meta_aircraft_type="A320",
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert result["meta_departure"][0] == "LFPG"
        assert result["meta_arrival"][0] == "EGLL"
        assert result["meta_aircraft_type"][0] == "A320"

    def test_identify_no_flightlist(self, tmp_path: Path) -> None:
        """No flightlist data → meta_departure/arrival/aircraft_type = null."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table_with_flightlist(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
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

        timestamps = [*range(0, 10), 38]
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[timestamps],
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        ids = result["meta_flight_id"].unique().to_list()
        assert len(ids) == 1

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
        "bds_tas_from_cas_kt": [],
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
            all_rows["bds_tas_from_cas_kt"].append(450.0)
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

    def test_derive_preserves_flag_columns(self, tmp_path: Path) -> None:
        """derive drops fdm_* but NOT fdm_flag_* columns."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir)

        # Add a fake fdm_flag_* column to the Delta Table
        df = pl.read_delta(str(table_path))
        df = df.with_columns(pl.lit(True).alias("fdm_flag_valid"))
        from node_fdm_data.delta import write_columns

        write_columns(df, table_path)

        config = self._make_config(tmp_path, data_dir=data_dir)
        derive(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert "fdm_flag_valid" in result.columns, "fdm_flag_valid was dropped by derive"


# ---------------------------------------------------------------------------
# Idempotency tests — identify (v3 étape 1)
# ---------------------------------------------------------------------------


class TestIdentifyIdempotent:
    """Idempotency tests for the ``identify`` command."""

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

    def test_identify_idempotent(self, tmp_path: Path) -> None:
        """Running identify twice produces no error and same flight IDs."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123", "abc123"],
            callsigns=["TEST01", "TEST01"],
            timestamps_s=[list(range(0, 20)), list(range(60, 80))],
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)
        first = pl.read_delta(str(table_path))

        identify(config=config, dry_run=False)
        second = pl.read_delta(str(table_path))

        assert (
            first["meta_flight_id"].sort().to_list() == second["meta_flight_id"].sort().to_list()
        )
        assert (
            first["meta_original_flight_id"].sort().to_list()
            == second["meta_original_flight_id"].sort().to_list()
        )

    def test_identify_fresh_vs_rerun(self, tmp_path: Path) -> None:
        """Re-run produces identical DataFrame to fresh run."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)
        first = pl.read_delta(str(data_dir / "flights.delta"))

        identify(config=config, dry_run=False)
        second = pl.read_delta(str(data_dir / "flights.delta"))

        # Compare all columns
        assert first.columns == second.columns
        for col in first.columns:
            assert first[col].to_list() == second[col].to_list(), f"{col} differs on re-run"

    def test_identify_partial_columns(self, tmp_path: Path) -> None:
        """Only meta_flight_id present (interrupted run) → still works."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        # Manually add only meta_flight_id (simulating partial/interrupted run)
        df = pl.read_delta(str(table_path))
        df = df.with_columns(pl.lit("partial_id").alias("meta_flight_id"))
        from node_fdm_data.delta import write_columns

        write_columns(df, table_path)

        config = self._make_config(tmp_path, data_dir=data_dir)
        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert "meta_flight_id" in result.columns
        assert result["meta_flight_id"][0] != "partial_id"


# ---------------------------------------------------------------------------
# Idempotency tests — flag (v3 étape 2)
# ---------------------------------------------------------------------------


def _make_identified_delta_table(tmp_path: Path, *, n_points: int = 50) -> Path:
    """Create a Delta Table with raw + identify columns (ready for flag step)."""
    from datetime import UTC, timedelta
    from datetime import datetime as dt

    base = dt(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    rows = {
        "raw_timestamp": [base + timedelta(seconds=i) for i in range(n_points)],
        "raw_icao24": ["abc123"] * n_points,
        "raw_callsign": ["TEST01"] * n_points,
        "raw_lat_deg": [48.0 + i * 0.01 for i in range(n_points)],
        "raw_lon_deg": [2.0 + i * 0.01 for i in range(n_points)],
        "raw_alt_ft": [35000.0] * n_points,
        "raw_gs_kt": [440.0] * n_points,
        "raw_track_deg": [90.0] * n_points,
        "raw_vz_ftmin": [100.0] * n_points,
        "meta_batch_date": ["20250101"] * n_points,
        "meta_original_flight_id": ["abc123_TEST01"] * n_points,
        "meta_flight_id": ["abc123_TEST01_s0"] * n_points,
    }
    df = pl.DataFrame(rows)
    table_path = tmp_path / "flights.delta"
    from node_fdm_data.delta import write_columns

    write_columns(df, table_path)
    return table_path


class TestFlagIdempotent:
    """Idempotency tests for the ``flag`` command."""

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

    def test_flag_idempotent(self, tmp_path: Path) -> None:
        """Running flag twice produces no error and same flag values."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_identified_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        flag(config=config, dry_run=False)
        first = pl.read_delta(str(table_path))

        flag(config=config, dry_run=False)
        second = pl.read_delta(str(table_path))

        flag_cols = [c for c in first.columns if c.startswith("fdm_flag_")]
        assert len(flag_cols) > 0, "No flag columns produced"
        for col in flag_cols:
            assert first[col].to_list() == second[col].to_list(), f"{col} changed on re-run"

    def test_flag_preserves_other_columns(self, tmp_path: Path) -> None:
        """Re-running flag does not touch era_* columns."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_identified_delta_table(data_dir)

        # Add fake era_* columns (as if enrich already ran)
        df = pl.read_delta(str(table_path))
        df = df.with_columns(
            pl.lit(280.0).alias("era_tas_kt"),
            pl.lit(0.78).alias("era_mach"),
        )
        from node_fdm_data.delta import write_columns

        write_columns(df, table_path)

        config = self._make_config(tmp_path, data_dir=data_dir)
        flag(config=config, dry_run=False)
        first = pl.read_delta(str(table_path))

        flag(config=config, dry_run=False)
        second = pl.read_delta(str(table_path))

        assert first["era_tas_kt"].to_list() == second["era_tas_kt"].to_list()
        assert first["era_mach"].to_list() == second["era_mach"].to_list()

    def test_flag_no_existing_no_drop(self, tmp_path: Path) -> None:
        """First run of flag (no existing flag cols) runs normally."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_identified_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        # Should not raise
        flag(config=config, dry_run=False)

        result = pl.read_delta(str(data_dir / "flights.delta"))
        assert "fdm_flag_valid" in result.columns


# ---------------------------------------------------------------------------
# Idempotency tests — segments (v3 étape 5)
# ---------------------------------------------------------------------------


class TestSegmentsIdempotent:
    """Idempotency tests for the ``segments`` command."""

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

    def test_segments_idempotent(self, tmp_path: Path) -> None:
        """Running segments twice produces no error and same sel columns."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir)

        # Derive first to get fdm_* columns needed by segments
        config = self._make_config(tmp_path, data_dir=data_dir)
        derive(config=config, dry_run=False)

        segments(config=config, dry_run=False)
        first = pl.read_delta(str(table_path))

        segments(config=config, dry_run=False)
        second = pl.read_delta(str(table_path))

        sel_cols = [c for c in first.columns if c.startswith("fdm_") and "_sel" in c]
        assert len(sel_cols) > 0, "No sel columns produced"
        for col in sel_cols:
            # NaN-aware comparison: fill NaN with sentinel then compare
            a = first[col].fill_nan(-999.0).fill_null(-999.0).to_list()
            b = second[col].fill_nan(-999.0).fill_null(-999.0).to_list()
            assert a == b, f"{col} changed on re-run"

    def test_segments_preserves_bds_sel_columns(self, tmp_path: Path) -> None:
        """Re-running segments does not drop bds_*_sel_* input columns."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir)

        config = self._make_config(tmp_path, data_dir=data_dir)
        derive(config=config, dry_run=False)

        segments(config=config, dry_run=False)
        first = pl.read_delta(str(table_path))

        segments(config=config, dry_run=False)
        second = pl.read_delta(str(table_path))

        # bds_mcp_sel_alt_ft is an input column that contains "_sel" — must be preserved
        assert "bds_mcp_sel_alt_ft" in first.columns
        assert "bds_mcp_sel_alt_ft" in second.columns
        assert first["bds_mcp_sel_alt_ft"].to_list() == second["bds_mcp_sel_alt_ft"].to_list()


# ---------------------------------------------------------------------------
# Idempotency tests — convert (v3 étapes 6-7)
# ---------------------------------------------------------------------------


class TestConvertIdempotent:
    """Idempotency tests for the ``convert`` command."""

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

    def test_convert_idempotent(self, tmp_path: Path) -> None:
        """Running convert twice produces no error and same SI/derivative values."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_derive_delta_table(data_dir)

        # Derive first to get fdm_* columns needed by convert
        config = self._make_config(tmp_path, data_dir=data_dir)
        derive(config=config, dry_run=False)

        convert(config=config, dry_run=False)
        first = pl.read_delta(str(table_path))

        convert(config=config, dry_run=False)
        second = pl.read_delta(str(table_path))

        si_cols = [c for c in first.columns if c.endswith(("_m", "_ms"))]
        deriv_cols = [c for c in first.columns if c.startswith("fdm_d_")]
        check_cols = [*si_cols, *deriv_cols]
        assert len(check_cols) > 0, "No SI/derivative columns produced"
        for col in check_cols:
            assert first[col].to_list() == second[col].to_list(), f"{col} changed on re-run"


# ---------------------------------------------------------------------------
# download mock tests (AC1)
# ---------------------------------------------------------------------------


class TestDownloadMock:
    """Tests for download with mocked OpenSky API."""

    @staticmethod
    def _make_config(tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "aircraft_db.csv").write_text(
            "icao24,registration,typecode,age,airline\nabc123,F-WXYZ,A320,5,AFR\n"
        )
        config = tmp_path / "config.yaml"
        config.write_text(f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n')
        return config

    def test_download_mock(self, tmp_path: Path) -> None:
        """Full download with mocked OpenSky writes Delta table columns."""
        from unittest.mock import MagicMock

        import pandas as pd

        config = self._make_config(tmp_path)

        history_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01 12:00", periods=5, freq="s"),
                "icao24": ["abc123"] * 5,
                "callsign": ["TEST01"] * 5,
                "latitude": [48.0] * 5,
                "longitude": [2.0] * 5,
                "altitude": [35000.0] * 5,
                "groundspeed": [440.0] * 5,
                "track": [90.0] * 5,
                "vertical_rate": [100.0] * 5,
            }
        )
        mock_history = MagicMock()
        mock_history.data = history_data

        mock_opensky = MagicMock()
        mock_opensky.history.return_value = mock_history
        mock_opensky.extended.return_value = None
        mock_opensky.flightlist.return_value = None

        mock_traffic_data = MagicMock()
        mock_traffic_data.opensky = mock_opensky

        written: list[pl.DataFrame] = []

        with (
            patch("node_fdm_pipeline.commands.data._require_traffic"),
            patch.dict(
                "sys.modules",
                {
                    "traffic": MagicMock(),
                    "traffic.core": MagicMock(),
                    "traffic.data": mock_traffic_data,
                },
            ),
            patch(
                "node_fdm_data.delta.write_columns",
                side_effect=lambda df, _p: written.append(df),
            ),
        ):
            download(
                config=config,
                start_date="2025-01-01",
                end_date="2025-01-02",
            )

        assert len(written) == 1
        df = written[0]
        raw_cols = [c for c in df.columns if c.startswith("raw_")]
        assert len(raw_cols) > 0
        assert "meta_batch_date" in df.columns
        assert "meta_departure" in df.columns
        assert "meta_arrival" in df.columns
        assert "meta_aircraft_type" in df.columns

    def test_download_empty_history(self, tmp_path: Path) -> None:
        """OpenSky returns None for history → no data written."""
        from unittest.mock import MagicMock

        config = self._make_config(tmp_path)

        mock_opensky = MagicMock()
        mock_opensky.history.return_value = None

        mock_traffic_data = MagicMock()
        mock_traffic_data.opensky = mock_opensky

        mock_write = MagicMock()

        with (
            patch("node_fdm_pipeline.commands.data._require_traffic"),
            patch.dict(
                "sys.modules",
                {
                    "traffic": MagicMock(),
                    "traffic.core": MagicMock(),
                    "traffic.data": mock_traffic_data,
                },
            ),
            patch("node_fdm_data.delta.write_columns", mock_write),
        ):
            download(
                config=config,
                start_date="2025-01-01",
                end_date="2025-01-02",
            )

        mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# preprocess mock tests (AC2)
# ---------------------------------------------------------------------------


class TestPreprocessPipeline:
    """Tests for the preprocess function with mocked dependencies."""

    @staticmethod
    def _make_config(tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        config = tmp_path / "config.yaml"
        config.write_text(f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n')
        return config

    def test_preprocess_pipeline(self, tmp_path: Path) -> None:
        """Preprocess reads delta, resamples, and writes back."""
        from datetime import UTC
        from datetime import datetime as dt

        config = self._make_config(tmp_path)

        n = 10
        df_input = pl.DataFrame(
            {
                "raw_timestamp": [dt(2025, 1, 1, 12, 0, i, tzinfo=UTC) for i in range(n)],
                "raw_icao24": ["abc123"] * n,
                "raw_callsign": ["TEST01"] * n,
                "raw_lat_deg": [48.0 + i * 0.001 for i in range(n)],
                "raw_lon_deg": [2.0] * n,
                "raw_alt_ft": [35000.0] * n,
                "meta_batch_date": ["20250101"] * n,
                "meta_flight_id": ["abc123_TEST01_s0"] * n,
            }
        )

        df_processed = df_input.head(5)

        with (
            patch("node_fdm_data.delta.read_delta_table", return_value=df_input),
            patch(
                "node_fdm_data.preprocessing.resample.preprocess_flights",
                return_value=df_processed,
            ),
        ):
            preprocess(config=config, dry_run=False)

        delta_table = tmp_path / "data" / "flights.delta"
        assert delta_table.exists()
        result = pl.read_delta(str(delta_table))
        assert len(result) == 5

    def test_preprocess_dry_run(self, tmp_path: Path) -> None:
        """Preprocess --dry-run validates config without modifying data."""
        config = self._make_config(tmp_path)
        preprocess(config=config, dry_run=True)

    def test_preprocess_rejects_missing_identify(self, tmp_path: Path) -> None:
        """Preprocess raises SystemExit when meta_flight_id is all null."""
        config = self._make_config(tmp_path)

        df_no_ids = pl.DataFrame(
            {
                "raw_timestamp": [None],
                "raw_icao24": ["abc123"],
                "meta_batch_date": ["20250101"],
                "meta_flight_id": [None],
            }
        )

        with (
            patch("node_fdm_data.delta.read_delta_table", return_value=df_no_ids),
            pytest.raises(SystemExit),
        ):
            preprocess(config=config, dry_run=False)

    def test_preprocess_rejects_no_flight_id_column(self, tmp_path: Path) -> None:
        """Preprocess raises SystemExit when meta_flight_id column is absent."""
        config = self._make_config(tmp_path)

        df_no_col = pl.DataFrame(
            {
                "raw_timestamp": [None],
                "raw_icao24": ["abc123"],
                "meta_batch_date": ["20250101"],
            }
        )

        with (
            patch("node_fdm_data.delta.read_delta_table", return_value=df_no_col),
            pytest.raises(SystemExit),
        ):
            preprocess(config=config, dry_run=False)


# ---------------------------------------------------------------------------
# convert mock tests (AC3)
# ---------------------------------------------------------------------------


class TestConvertToSI:
    """Tests for the convert function with mocked dependencies."""

    @staticmethod
    def _make_config(tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        config = tmp_path / "config.yaml"
        config.write_text(f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n')
        return config

    def test_convert_pipeline(self, tmp_path: Path) -> None:
        """Convert reads delta, applies SI + derivatives, writes back."""
        config = self._make_config(tmp_path)

        df_input = pl.DataFrame(
            {
                "raw_alt_ft": [35000.0, 35100.0],
                "raw_gs_kt": [440.0, 441.0],
                "meta_batch_date": ["20250101", "20250101"],
            }
        )

        df_si = df_input.with_columns(pl.col("raw_alt_ft").alias("raw_alt_m"))
        df_deriv = df_si.with_columns(pl.lit(0.0).alias("fdm_d_alt_ms"))

        written: list[pl.DataFrame] = []

        with (
            patch("node_fdm_data.delta.read_delta_table", return_value=df_input),
            patch("node_fdm_data.preprocessing.convert.convert_si", return_value=df_si),
            patch(
                "node_fdm_data.preprocessing.convert.compute_derivatives",
                return_value=df_deriv,
            ),
            patch(
                "node_fdm_data.delta.write_columns",
                side_effect=lambda df, _p: written.append(df),
            ),
            patch("node_fdm_data.preprocessing.convert.SI_CONVERSIONS", []),
            patch("node_fdm_data.preprocessing.convert.SI_DERIVATIVES", []),
        ):
            convert(config=config, dry_run=False)

        assert len(written) == 1
        result = written[0]
        assert "raw_alt_m" in result.columns
        assert "fdm_d_alt_ms" in result.columns

    def test_convert_empty_table(self, tmp_path: Path) -> None:
        """Convert with empty delta table processes without crash."""
        config = self._make_config(tmp_path)

        df_empty = pl.DataFrame(
            {
                "raw_alt_ft": pl.Series([], dtype=pl.Float64),
                "meta_batch_date": pl.Series([], dtype=pl.Utf8),
            }
        )

        written: list[pl.DataFrame] = []

        with (
            patch("node_fdm_data.delta.read_delta_table", return_value=df_empty),
            patch("node_fdm_data.preprocessing.convert.convert_si", return_value=df_empty),
            patch(
                "node_fdm_data.preprocessing.convert.compute_derivatives",
                return_value=df_empty,
            ),
            patch(
                "node_fdm_data.delta.write_columns",
                side_effect=lambda df, _p: written.append(df),
            ),
            patch("node_fdm_data.preprocessing.convert.SI_CONVERSIONS", []),
            patch("node_fdm_data.preprocessing.convert.SI_DERIVATIVES", []),
        ):
            convert(config=config, dry_run=False)

        assert len(written) == 1
        assert len(written[0]) == 0

    def test_convert_dry_run(self, tmp_path: Path) -> None:
        """Convert --dry-run validates config without modifying data."""
        config = self._make_config(tmp_path)
        convert(config=config, dry_run=True)


# ---------------------------------------------------------------------------
# clean-speeds tests
# ---------------------------------------------------------------------------


def _make_clean_speeds_delta_table(tmp_path: Path) -> Path:
    """Create a Delta Table seeded with bds_* and era_* columns."""
    import math
    from datetime import UTC, timedelta
    from datetime import datetime as dt

    import numpy as np
    from node_fdm_data.delta import write_columns

    n = 30
    bds_mach = np.full(n, 0.80)
    bds_ias_kt = np.full(n, 250.0)
    bds_tas_kt = np.full(n, 230.0)
    bds_mach[15] = 1.5
    bds_ias_kt[15] = 800.0
    bds_tas_kt[15] = 800.0

    base = dt(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    # raw_alt_ft above on-ground threshold so the cleaning pipeline does
    # not blank out the entire fixture; raw_vz_ftmin = 0 with high alt
    # still counts as airborne (mask requires both alt<1500 AND |vz|<200).
    df = pl.DataFrame(
        {
            "raw_timestamp": [base + timedelta(seconds=i) for i in range(n)],
            "raw_icao24": ["abc123"] * n,
            "meta_flight_id": ["abc123_TST_s0"] * n,
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
    table_path = tmp_path / "flights.delta"
    write_columns(df, table_path)
    _ = math  # silence unused-import lint when math not used elsewhere
    return table_path


class TestCleanSpeedsCommand:
    """Integration tests for the ``clean-speeds`` pipeline stage."""

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

    def test_clean_speeds_stage_writes_clean_columns(self, tmp_path: Path) -> None:
        """Stage writes bds_*_clean columns to the Delta Table."""
        from node_fdm_pipeline.commands.data import clean_speeds

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_clean_speeds_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        clean_speeds(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        for col in ("bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean"):
            assert col in result.columns, f"Missing column: {col}"

    def test_clean_speeds_stage_idempotent_on_table(self, tmp_path: Path) -> None:
        """Running the stage twice produces identical clean columns."""
        import math

        from node_fdm_pipeline.commands.data import clean_speeds

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_clean_speeds_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        clean_speeds(config=config, dry_run=False)
        first = pl.read_delta(str(table_path)).sort("raw_timestamp")

        clean_speeds(config=config, dry_run=False)
        second = pl.read_delta(str(table_path)).sort("raw_timestamp")

        for col in ("bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean"):
            a = first[col].to_list()
            b = second[col].to_list()
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=False):
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    assert y is None or (isinstance(y, float) and math.isnan(y))
                else:
                    assert x == pytest.approx(y)

    def test_clean_speeds_stage_preserves_raw_bds(self, tmp_path: Path) -> None:
        """Raw bds_* columns are unchanged in the Delta Table after the stage."""
        from node_fdm_pipeline.commands.data import clean_speeds

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_clean_speeds_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        before = pl.read_delta(str(table_path)).sort("raw_timestamp")
        raw_mach = before["bds_mach"].to_list()
        raw_ias = before["bds_ias_kt"].to_list()
        raw_tas = before["bds_tas_kt"].to_list()

        clean_speeds(config=config, dry_run=False)

        after = pl.read_delta(str(table_path)).sort("raw_timestamp")
        assert after["bds_mach"].to_list() == raw_mach
        assert after["bds_ias_kt"].to_list() == raw_ias
        assert after["bds_tas_kt"].to_list() == raw_tas

    def test_clean_speeds_stage_dry_run_no_write(self, tmp_path: Path) -> None:
        """`--dry-run` validates config and does not write clean columns."""
        from node_fdm_pipeline.commands.data import clean_speeds

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        table_path = _make_clean_speeds_delta_table(data_dir)
        config = self._make_config(tmp_path, data_dir=data_dir)

        clean_speeds(config=config, dry_run=True)

        result = pl.read_delta(str(table_path))
        assert "bds_mach_clean" not in result.columns
