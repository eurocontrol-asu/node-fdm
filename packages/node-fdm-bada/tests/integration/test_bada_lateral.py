from __future__ import annotations

import itertools
import math
from pathlib import Path

import polars as pl
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def bada_aircraft() -> object:
    py_bada = pytest.importorskip("pyBADA.bada4")
    import os

    bada_dir = os.environ.get("BADA_4_2_DIR")
    if not bada_dir or not Path(bada_dir).exists():
        pytest.skip("BADA_4_2_DIR not set or missing")
    try:
        return py_bada.Bada4Aircraft("4.2", filePath=bada_dir, acName="A320-231")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BADA aircraft load failed: {exc}")


def test_real_step_produces_lat_lon_progression(bada_aircraft: object, tmp_path: Path) -> None:
    from node_fdm_bada.predictor import process_single_flight

    n = 5
    df = pl.DataFrame(
        {
            "alt_std_m": [10000.0] * n,
            "tas_ms": [250.0] * n,
            "temperature": [223.0] * n,
            "mach": [0.78] * n,
            "cas_ms": [150.0] * n,
            "cas_sel_ms": [150.0] * n,
            "mach_sel": [0.0] * n,
            "vz_sel_ms": [0.0] * n,
            "alt_sel_m": [10000.0] * n,
            "long_wind_ms": [0.0] * n,
            "raw_lat_deg": [43.0] * n,
            "raw_lon_deg": [2.0] * n,
            "fdm_heading_rad": [math.pi / 2] * n,
            "fdm_heading_target_rad": [math.pi / 2] * n,
            "fdm_heading_target_known": [True] * n,
        }
    )
    p = tmp_path / "flight.parquet"
    df.write_parquet(p)

    out = process_single_flight(p, bada_aircraft)

    assert out is not None
    assert "bada_lat_deg" in out.columns
    assert "bada_lon_deg" in out.columns
    assert "bada_heading_rad" in out.columns

    lons = out["bada_lon_deg"].to_list()
    assert all(b > a for a, b in itertools.pairwise(lons)), (
        f"longitude not monotonically increasing eastward: {lons}"
    )
    headings = out["bada_heading_rad"].to_list()
    for h in headings:
        assert h == pytest.approx(math.pi / 2, abs=0.05)
