"""E2E: ``fdm derive --config <yaml>`` honours the lateral_detection block.

Writes a tiny preprocessed Delta fixture, points a YAML at it with an
over-the-top ``rate_threshold``, and asserts that the resulting derived
Delta has ``fdm_in_turn`` False everywhere (the high threshold suppresses
all detections).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.e2e


def _write_preprocessed_delta(out_dir: Path) -> None:
    """Single short turning flight, written as a Delta table."""
    n = 100
    track = np.concatenate(
        [
            np.full(40, 0.0),
            np.linspace(0.0, 90.0, 20),
            np.full(40, 90.0),
        ]
    )
    df = pl.DataFrame(
        {
            "flight_id": np.zeros(n, dtype=np.int64),
            "typecode": ["A320"] * n,
            "latitude": np.linspace(45.0, 46.0, n),
            "longitude": np.linspace(2.0, 3.0, n),
            "track": track,
            "timestamp": np.arange(n, dtype=np.int64),
        }
    )
    df.write_delta(str(out_dir))


def test_derive_honours_yaml_lateral_block(tmp_path: Path) -> None:
    """AC5/AC6: a high rate_threshold in YAML suppresses fdm_in_turn detections."""
    data_dir = tmp_path / "data"
    preprocess_dir = data_dir / "preprocessed_parquet"
    preprocess_dir.mkdir(parents=True)
    _write_preprocessed_delta(preprocess_dir / "A320")

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""\
            paths:
              data_dir: {data_dir}
            typecodes:
              - A320
            lateral_detection:
              rate_threshold: 10.0
            """
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "node_fdm_pipeline",
            "derive",
            "--config",
            str(cfg_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    derived_dir = data_dir / "processed_flights" / "A320"
    derived = pl.read_delta(str(derived_dir))
    assert not derived["fdm_in_turn"].to_numpy().any()
