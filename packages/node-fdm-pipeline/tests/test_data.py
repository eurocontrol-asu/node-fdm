"""Tests for data pipeline commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import polars as pl
import pytest

from node_fdm_pipeline.commands.data import (
    _require_traffic,
    aircraft_list,
    download,
    process,
)

if TYPE_CHECKING:
    pass


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

    def test_process_empty_preprocess_dir(self, tmp_path: Path) -> None:
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

        process(arch="opensky", config=config, dry_run=False)

        # No output files — clean return on empty dir
        assert not process_dir.exists() or len(list(process_dir.iterdir())) == 0

    def test_process_single_file(self, tmp_path: Path) -> None:
        """Process a synthetic parquet and verify output exists."""
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

        # Create a synthetic parquet with columns flight_processing expects
        df = pl.DataFrame(
            {
                "flight_id": ["F001"] * 10,
                "timestamp": list(range(10)),
                "altitude_ft": [35000.0 + i * 100 for i in range(10)],
                "alt_sel_ft": [35000.0] * 10,
                "vz_sel_ftmin": [0.0] * 10,
                "mach_sel": [0.82] * 10,
                "cas_sel_kt": [280.0] * 10,
                "groundspeed": [450.0 + i for i in range(10)],
                "vertical_rate": [0.0] * 10,
                "latitude": [48.0 + i * 0.01 for i in range(10)],
                "longitude": [2.0 + i * 0.01 for i in range(10)],
                "track": [90.0] * 10,
                "typecode": ["A320"] * 10,
                "icao24": ["abc123"] * 10,
            }
        )
        df.write_parquet(preprocess_dir / "processed_20250101.parquet")

        # Process — flight_processing may filter out short segments,
        # but should still produce an output file.
        # Mock split_by_icao since file naming doesn't match expected convention
        mock_split = pl.DataFrame(
            {"filepath": ["processed_20250101.parquet"], "icao": ["A320"], "split": ["train"]}
        )
        with patch("node_fdm_data.split.split_by_icao", return_value=mock_split):
            process(arch="opensky", config=config, dry_run=False)

        output = process_dir / "processed_20250101.parquet"
        assert output.exists()

    def test_process_skip_existing(self, tmp_path: Path) -> None:
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

        # Create input AND output so it should skip
        df = pl.DataFrame({"x": [1, 2, 3]})
        df.write_parquet(preprocess_dir / "file.parquet")
        df.write_parquet(process_dir / "file.parquet")

        # Should not raise — skip with info log
        process(arch="opensky", config=config, dry_run=False)


class TestDownloadCommand:
    """Tests for the ``download`` command."""

    def test_download_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without writing files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "aircraft_db.csv").write_text(
            "icao24,registration,typecode,age,airline\n" "abc123,F-WXYZ,A320,5,AFR\n"
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
  data_dir: "{tmp_path / 'data'}"

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
