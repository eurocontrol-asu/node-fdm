from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.integration


def test_real_turn_segment_produces_progressive_heading_change(tmp_path: Path) -> None:
    """AC5: bada_heading_rad evolves monotonically over a real turn segment."""
    pytest.importorskip("pyBADA")
    try:
        from pyBADA.bada4 import Bada4Aircraft

        ac = Bada4Aircraft(badaVersion="4.2", acName="A320-251N")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"pyBADA Bada4Aircraft unavailable: {exc}")

    from node_fdm_bada.predictor import process_single_flight

    n = 60
    dt = 1.0
    turn_rate_rads = math.radians(3.0)
    initial_heading_rad = math.radians(90.0)

    headings = [initial_heading_rad + turn_rate_rads * i * dt for i in range(n)]

    df = pl.DataFrame(
        {
            "alt_std_m": [10000.0] * n,
            "tas_ms": [200.0] * n,
            "temperature": [223.15] * n,
            "mach_sel": [0.0] * n,
            "alt_sel_m": [10000.0] * n,
            "vz_sel_ms": [0.0] * n,
            "long_wind_ms": [0.0] * n,
            "mach": [0.78] * n,
            "cas_sel_ms": [130.0] * n,
            "raw_lat_deg": [45.0] * n,
            "raw_lon_deg": [5.0] * n,
            "fdm_heading_rad": headings,
            "fdm_heading_target_rad": headings,
            "fdm_heading_target_known": [True] * n,
            "fdm_in_turn": [True] * n,
            "fdm_d_heading_rads": [turn_rate_rads] * n,
        }
    )

    flight_path = tmp_path / "turn.parquet"
    df.write_parquet(flight_path)

    result = process_single_flight(flight_path, ac)

    assert result is not None
    assert "bada_heading_rad" in result.columns

    headings_out = np.unwrap(np.asarray(result["bada_heading_rad"].to_list()))
    diffs = np.diff(headings_out)
    assert (diffs > 0).sum() / len(diffs) > 0.8

    total_change = headings_out[-1] - headings_out[0]
    expected_change = turn_rate_rads * dt * (n - 1)
    assert abs(total_change - expected_change) / expected_change < 0.10
