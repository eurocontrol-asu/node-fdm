"""Integration tests for the segments pipeline stage.

Real Delta Table I/O for a seeded synthetic flight, exercising
fdm_pipeline.commands.data.segments end-to-end at the table boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from node_fdm_pipeline.config import SelectedParamConfig

pytestmark = pytest.mark.integration


_FT_TO_M = 0.3048


def _seed_flight_df(*, with_tail_gap: bool = False) -> pl.DataFrame:
    n_climb, n_cruise, n_descent = 100, 100, 100
    n_tail = 50 if with_tail_gap else 0
    n = n_climb + n_cruise + n_descent + n_tail
    alt = np.empty(n)
    alt[:n_climb] = np.linspace(1000.0, 34000.0, n_climb)
    alt[n_climb : n_climb + n_cruise] = 34000.0
    alt[n_climb + n_cruise : n_climb + n_cruise + n_descent] = np.linspace(
        34000.0, 1000.0, n_descent
    )
    if with_tail_gap:
        alt[-n_tail:] = np.nan
    mach = np.full(n, np.nan)
    mach[n_climb : n_climb + n_cruise] = 0.78
    cas = np.full(n, np.nan)
    cas[:n_climb] = 280.0
    cas[n_climb + n_cruise : n_climb + n_cruise + n_descent] = 270.0
    return pl.DataFrame(
        {
            "flight_id": ["F1"] * n,
            "time_idx": np.arange(n, dtype=np.int64),
            "era_mach": mach,
            "bds_ias_kt": cas,
            "raw_alt_ft": alt,
            "raw_vz_ftmin": np.zeros(n),
        }
    )


def _run_segments_stage(
    input_path: Path,
    output_path: Path,
    selected_params: SelectedParamConfig,
) -> None:
    """Invoke the pipeline segments stage on Delta tables."""
    pytest.importorskip("deltalake")
    from node_fdm_pipeline.commands.data import segments

    segments.run(  # type: ignore[attr-defined]
        input_path=str(input_path),
        output_path=str(output_path),
        selected_params=selected_params,
    )


def test_segments_stage_emits_known_mask(
    tmp_path: Path, selected_param_config_factory: Callable[[], SelectedParamConfig]
) -> None:
    deltalake = pytest.importorskip("deltalake")
    input_path = tmp_path / "input.delta"
    output_path = tmp_path / "output.delta"

    df = _seed_flight_df()
    deltalake.write_deltalake(str(input_path), df.to_arrow())

    _run_segments_stage(input_path, output_path, selected_param_config_factory())

    out = pl.read_delta(str(output_path))
    assert "fdm_tas_target_known" in out.columns


def test_segments_stage_no_global_backfill_on_disk(
    tmp_path: Path, selected_param_config_factory: Callable[[], SelectedParamConfig]
) -> None:
    deltalake = pytest.importorskip("deltalake")
    input_path = tmp_path / "input.delta"
    output_path = tmp_path / "output.delta"

    df = _seed_flight_df(with_tail_gap=True)
    deltalake.write_deltalake(str(input_path), df.to_arrow())

    _run_segments_stage(input_path, output_path, selected_param_config_factory())

    out = pl.read_delta(str(output_path))
    target = out["fdm_tas_target_kt"].to_numpy()
    # Trailing rows (where altitude was NaN — no segment) must remain NaN
    assert np.all(np.isnan(target[-50:]))
