from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest

from _config_fixtures import SELECTED_PARAMS_YAML

NAN: float = float("nan")


def _seed_delta(path: Path) -> None:
    pl.DataFrame(
        {
            "meta_flight_id": ["F1"] * 4,
            "fdm_in_turn": [False] * 4,
            "fdm_alt_sel_ft": [10000.0] * 4,
            "fdm_vz_sel_ftmin": [NAN] * 4,
            "fdm_gamma_sel_rad": [NAN] * 4,
            "fdm_mach_sel": [0.78] * 4,
            "fdm_cas_sel_kt": [250.0, 252.0, 254.0, 256.0],
        },
        schema={
            "meta_flight_id": pl.Utf8,
            "fdm_in_turn": pl.Boolean,
            "fdm_alt_sel_ft": pl.Float64,
            "fdm_vz_sel_ftmin": pl.Float64,
            "fdm_gamma_sel_rad": pl.Float64,
            "fdm_mach_sel": pl.Float64,
            "fdm_cas_sel_kt": pl.Float64,
        },
    ).write_delta(str(path))


def _write_config(config_path: Path, data_dir: Path) -> None:
    config_path.write_text(
        f"paths:\n  data_dir: {data_dir}\ntypecodes:\n  - A320\n" + SELECTED_PARAMS_YAML
    )


@pytest.mark.e2e
def test_fdm_label_modes_dry_run_exits_zero(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    delta_path = data_dir / "flights.delta"
    config_path = tmp_path / "config.yaml"
    _seed_delta(delta_path)
    _write_config(config_path, data_dir)

    result = subprocess.run(
        [
            "uv",
            "run",
            "fdm",
            "label-modes",
            "--config",
            str(config_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "label_modes_dry_run" in (result.stdout + result.stderr)


@pytest.mark.e2e
def test_fdm_label_modes_writes_column_to_delta(tmp_path: Path) -> None:
    from node_fdm_data.delta import read_delta_table

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    delta_path = data_dir / "flights.delta"
    config_path = tmp_path / "config.yaml"
    _seed_delta(delta_path)
    _write_config(config_path, data_dir)

    subprocess.run(
        ["uv", "run", "fdm", "label-modes", "--config", str(config_path)],
        check=True,
    )
    out = read_delta_table(delta_path)
    assert "fdm_mode_label" in out.columns
