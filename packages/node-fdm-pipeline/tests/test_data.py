"""Tests for data pipeline commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import polars as pl
import pytest

from node_fdm_pipeline.commands.data import (
    _require_traffic,
    _split_at_gaps,
    aircraft_list,
    download,
    process,
)

if TYPE_CHECKING:
    from traffic.core import Flight


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


class TestProcessCommand:
    """Tests for the ``process`` command (no traffic dependency)."""

    def test_process_dry_run(self, tmp_config: Path) -> None:
        """--dry-run validates config without writing files."""
        process(arch="opensky", config=tmp_config, dry_run=True)
        # No files should be created beyond the config dir

    def test_process_empty_preprocess_dir(self, tmp_path: Path, mocker: Any) -> None:
        """Warning log and clean return when no parquet files found."""
        # Create a config pointing at an empty preprocess dir
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"

typecodes:
  - A320
"""
        )

        mock_arco_cls = mocker.MagicMock()
        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        process(arch="opensky", config=config, dry_run=False)

        # No output files — clean return on empty dir
        assert not process_dir.exists() or len(list(process_dir.iterdir())) == 0

    def test_process_single_file(self, tmp_path: Path, mocker: Any) -> None:
        """Process a synthetic parquet and verify output exists."""
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)
        era5_cache = data_dir / "era5_cache"
        era5_cache.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"
  era5_cache_dir: "era5_cache"

typecodes:
  - A320
"""
        )

        n = 50
        # Create a synthetic parquet with all required columns
        df = pl.DataFrame(
            {
                "flight_id": ["F001"] * n,
                "timestamp": [float(i * 4) for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "selected_mcp": [35000.0] * n,
                "vertical_rate": [100.0] * n,
                "Mach": [0.78] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                "latitude": [48.0 + i * 0.001 for i in range(n)],
                "longitude": [2.0 + i * 0.001 for i in range(n)],
                "track": [90.0] * n,
                "heading": [88.0] * n,
                "typecode": ["A320"] * n,
                "icao24": ["abc123"] * n,
                "adep_dist": [100.0 - i for i in range(n)],
                "ades_dist": [float(i * 2) for i in range(n)],
            }
        )
        df.write_parquet(preprocess_dir / "processed_20250101.parquet")

        # Mock fastmeteo — interpolate adds weather columns
        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            pdf["temperature"] = 220.0
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        mock_arco_cls = mocker.MagicMock()
        mock_arco_instance = mocker.MagicMock()
        mock_arco_instance.interpolate.side_effect = fake_interpolate
        mock_arco_cls.return_value = mock_arco_instance

        # Patch the full import chain: fastmeteo.source.arco_era5.ArcoEra5
        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        process(arch="opensky", config=config, dry_run=False)

        output = process_dir / "processed_20250101.parquet"
        assert output.exists()

        # Verify output has the new derived columns
        result = pl.read_parquet(output)
        assert "gamma_air" in result.columns
        assert "long_wind" in result.columns
        assert "mach_sel" in result.columns
        assert "distance_along_track_m" in result.columns
        # Verify lateral augmentation columns
        assert "track_ortho" in result.columns
        assert "drift_angle" in result.columns

    def test_process_skip_existing(self, tmp_path: Path, mocker: Any) -> None:
        """Already-processed files are skipped."""
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"

typecodes:
  - A320
"""
        )

        mock_arco_cls = mocker.MagicMock()
        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        # Create input AND output so it should skip
        df = pl.DataFrame({"x": [1, 2, 3]})
        df.write_parquet(preprocess_dir / "file.parquet")
        df_out = pl.DataFrame({"flight_id": ["F1", "F2"], "typecode": ["A320", "A320"]})
        df_out.write_parquet(process_dir / "file.parquet")

        # Should not raise — skip with info log
        process(arch="opensky", config=config, dry_run=False)


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

    def test_preprocess_help(self) -> None:
        """fdm preprocess --help shows expected flags."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "preprocess", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--history-file" in output
        assert "--workers" in output
        assert "--dry-run" in output

    def test_process_help_has_dry_run(self) -> None:
        """fdm process --help now shows --dry-run flag."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "process", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--dry-run" in output


# ---------------------------------------------------------------------------
# _split_at_gaps tests (requires traffic)
# ---------------------------------------------------------------------------

traffic = pytest.importorskip("traffic")


def _make_flight(n_points: int, *, gap_at: int | None = None, gap_seconds: int = 60) -> Flight:
    """Create a synthetic Flight with optional data gap.

    Args:
        n_points: Total number of data points.
        gap_at: Index at which to insert a gap.  Points before ``gap_at``
            are 1 s apart; after the gap the first timestamp jumps by
            ``gap_seconds``.
        gap_seconds: Duration of the gap in seconds.
    """
    import numpy as np
    import pandas as pd
    from traffic.core import Flight

    # Build timestamps: 1s apart, with optional gap
    times = []
    t0 = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
    for i in range(n_points):
        if gap_at is not None and i == gap_at:
            t0 += pd.Timedelta(seconds=gap_seconds)
        times.append(t0)
        t0 += pd.Timedelta(seconds=1)

    return Flight(
        pd.DataFrame(
            {
                "timestamp": times,
                "icao24": ["abc123"] * n_points,
                "callsign": ["TEST01"] * n_points,
                "latitude": np.linspace(48.0, 49.0, n_points),
                "longitude": np.linspace(2.0, 3.0, n_points),
                "altitude": np.full(n_points, 35000.0),
            }
        )
    )


class TestSplitAtGaps:
    """Tests for the ``_split_at_gaps`` helper function."""

    def test_split_no_gap(self) -> None:
        """Continuous flight → 1 segment with original_flight_id set."""
        from traffic.core import Traffic

        f = _make_flight(60)
        t = Traffic.from_flights([f])
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is not None
        assert len(result) == 1
        seg = next(iter(result))
        assert "original_flight_id" in seg.data.columns
        assert seg.data["original_flight_id"].iloc[0] == "abc123_TEST01"

    def test_split_with_gap(self) -> None:
        """Flight with 60s gap → 2 segments, both with same original_flight_id."""
        from traffic.core import Traffic

        f = _make_flight(80, gap_at=40, gap_seconds=60)
        t = Traffic.from_flights([f])
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is not None
        assert len(result) == 2
        segments = list(result)
        # Both segments share the same original flight identity
        ids = {seg.data["original_flight_id"].iloc[0] for seg in segments}
        assert ids == {"abc123_TEST01"}

    def test_split_short_segments_filtered(self) -> None:
        """Segments shorter than min_points are discarded."""
        from traffic.core import Traffic

        # 50 pts total, gap at 5 → seg[0]=5 pts (too short), seg[1]=45 pts (ok)
        f = _make_flight(50, gap_at=5, gap_seconds=60)
        t = Traffic.from_flights([f])
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is not None
        assert len(result) == 1  # only the 45-pt segment survives

    def test_split_all_filtered_returns_none(self) -> None:
        """All segments too short → returns None."""
        from traffic.core import Traffic

        # 10 pts, gap at 5 → two segments of 5 pts each, both below min_points=20
        f = _make_flight(10, gap_at=5, gap_seconds=60)
        t = Traffic.from_flights([f])
        result = _split_at_gaps(t, threshold="30s", min_points=20)

        assert result is None

    def test_split_empty_traffic(self) -> None:
        """Traffic with only a tiny flight → returns None (all segments too short)."""
        from traffic.core import Traffic

        # 3 points — always below any reasonable min_points
        f = _make_flight(3)
        t = Traffic.from_flights([f])
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is None
