"""Black-box campaign chain from recorded acquisition through selection-bounded identify."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from _config_fixtures import write_config
from node_fdm_pipeline.commands import _raw_cache as raw_cache
from node_fdm_pipeline.config import PipelineConfig

pytestmark = pytest.mark.e2e


def _selection_rows() -> list[dict[str, object]]:
    alpha: dict[str, object] = {
        "icao24": "a00001",
        "callsign": "ALPHA1",
        "firstseen": 1577835000,
        "lastseen": 1577838600,
        "msn": "M1",
        "split": "train",
        "selection_id": "sel-alpha",
        "utc_days": ["20200101"],
        "acquisition_key": "acq-alpha",
    }
    return [
        {**alpha, "cohort": "C1"},
        {**alpha, "cohort": "C2"},
        {
            "icao24": "z00002",
            "callsign": "ZULU2",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "msn": "M2",
            "split": "test",
            "cohort": "C1",
            "selection_id": "sel-zulu",
            "utc_days": ["20200101"],
            "acquisition_key": "acq-zulu",
        },
    ]


def _write_fleet(tmp_path: Path) -> tuple[Path, Path, Path]:
    fleet_dir = tmp_path / "fleet"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "aircraft_db.csv").write_text(
        "icao24,registration,typecode,age,airline\n"
        "a00001,F-ALFA,A320,5,AAA\n"
        "z00002,F-ZULU,A320,5,ZZZ\n"
        "b00003,F-BETA,A320,5,BBB\n"
        "c00004,F-CHAR,A320,5,CCC\n",
        encoding="utf-8",
    )
    for cohort, parasite in (("C1", "b00003"), ("C2", "c00004")):
        cohort_dir = fleet_dir / "types" / cohort
        results_dir = cohort_dir / "results"
        results_dir.mkdir(parents=True)
        write_config(cohort_dir / "config.yaml", data_dir)
        (results_dir / f"selection_{cohort}.csv").write_text(
            f"icao24,day\na00001,2020-01-01\n{parasite},2020-01-01\n",
            encoding="utf-8",
        )

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(_selection_rows()), encoding="utf-8")
    return fleet_dir, data_dir, selection


def _write_recorded_cache(data_dir: Path) -> None:
    from datetime import UTC, datetime

    history = pl.DataFrame(
        [
            {
                "timestamp": datetime.fromtimestamp(1577835000, tz=UTC),
                "icao24": "a00001",
                "callsign": "  alpha1  ",
                "latitude": 48.0,
                "longitude": 2.0,
                "altitude": 35000.0,
                "groundspeed": 440.0,
                "track": 90.0,
                "vertical_rate": 100.0,
                "firstseen": 1577835000,
                "lastseen": 1577838600,
            },
            {
                "timestamp": datetime.fromtimestamp(1577900000, tz=UTC),
                "icao24": "a00001",
                "callsign": "  alpha1  ",
                "latitude": 49.0,
                "longitude": 3.0,
                "altitude": 34000.0,
                "groundspeed": 430.0,
                "track": 91.0,
                "vertical_rate": -100.0,
                "firstseen": 1577900000,
                "lastseen": 1577903600,
            },
        ]
    )
    cache_path = data_dir / "raw" / "history" / "date=20200101" / "icao24=a00001" / "data.parquet"
    cache_path.parent.mkdir(parents=True)
    history.write_parquet(cache_path)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_download_decode_identify_chain_is_bounded_by_selection(tmp_path: Path) -> None:
    """AC4: the three CLI façades yield one alpha and a durable absent zulu rejection."""
    fleet_dir, data_dir, selection = _write_fleet(tmp_path)
    _write_recorded_cache(data_dir)
    fdm = Path(sys.executable).with_name("fdm")

    download = _run(
        [
            str(fdm),
            "download-fleet",
            "--fleet-dir",
            str(fleet_dir),
            "--dry-run",
        ]
    )
    decode = _run(
        [
            str(fdm),
            "decode-fleet",
            "--fleet-dir",
            str(fleet_dir),
        ]
    )
    config = fleet_dir / "types" / "C1" / "config.yaml"
    identify = _run(
        [
            str(fdm),
            "identify",
            "--config",
            str(config),
            "--selection",
            str(selection),
        ]
    )

    assert download.returncode == 0, download.stderr
    assert decode.returncode == 0, decode.stderr
    assert identify.returncode == 0, identify.stderr

    identified = pl.read_delta(str(data_dir / "flights.delta"))
    alpha = identified.filter(pl.col("selection_id") == "sel-alpha")
    assert alpha.height == 1
    assert alpha["firstseen"].to_list() == [1577835000]
    assert 1577900000 not in identified["firstseen"].to_list()

    cfg = PipelineConfig.from_yaml(config)
    receipt = raw_cache.read_absence_receipt(
        raw_cache.cache_root(cfg, "history"),
        "20200101",
        "history",
    )
    assert receipt is not None
    assert "sel-zulu" in receipt.icao24s
