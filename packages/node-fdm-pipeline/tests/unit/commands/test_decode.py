"""Unit tests for ``decode`` filter logic (AXM-1681, AC5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from node_fdm_pipeline.config import PipelineConfig


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    from node_fdm_pipeline.config import PipelineConfig

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "aircraft_db.csv").write_text(
        "icao24,registration,typecode,age,airline\n"
        "a,reg,A320,5,AFR\n"
        "b,reg,A320,5,AFR\n"
        "c,reg,A320,5,AFR\n"
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""
    )
    return PipelineConfig.from_yaml(cfg_path)


def test_decode_filter_intersects_with_csv(
    cfg: PipelineConfig,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC5: demanded set = aircraft_db.csv ∩ icao24-filter file.

    With csv=[a,b,c] and filter=[b,c,d], the decoded set must be {b, c}.
    Cache reads return empty, so the function exits early and only the
    demanded-set computation is exercised. Spy on ``_raw_cache.read_partition``
    to capture which icao24 lists were requested.
    """
    from node_fdm_pipeline.commands import data as data_mod

    requested: list[set[str]] = []

    def fake_read_partition(_cfg, _kind, _date, icao24_list, **_kwargs):  # type: ignore[no-untyped-def]
        requested.append(set(icao24_list))
        return pl.DataFrame()

    mocker.patch(
        "node_fdm_pipeline.commands._raw_cache.read_partition",
        side_effect=fake_read_partition,
    )
    # Make is_cached / read_parquet noop-ish so flightlist branch returns nothing
    mocker.patch(
        "node_fdm_pipeline.commands._raw_cache.is_cached",
        return_value=False,
    )

    filter_file = tmp_path / "filter.txt"
    filter_file.write_text("b\nc\nd\n")

    cfg_path = tmp_path / "config.yaml"
    # Re-derive the same cfg path used by the fixture
    for p in tmp_path.glob("config.yaml"):
        cfg_path = p

    data_mod.decode(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
        icao24_filter=filter_file,
    )

    assert requested, "decode should request at least one cache partition"
    demanded = set().union(*requested)
    assert demanded == {"b", "c"}
