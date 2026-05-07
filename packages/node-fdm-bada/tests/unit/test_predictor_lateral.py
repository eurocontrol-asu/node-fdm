from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from node_fdm_bada import predictor as predictor_mod
from node_fdm_bada.predictor import _run_bada_step, process_single_flight


class _FakeAC:
    MTOW = 70000.0


def _make_recorder() -> tuple[list[dict[str, Any]], Any]:
    """Returns (calls_list, sentinel_factory).

    Each TCL stub records its kwargs and returns a pandas-like DataFrame
    with the columns the predictor reads.
    """
    import pandas as pd

    calls: list[dict[str, Any]] = []

    def make_stub(name: str) -> Any:
        def stub(**kwargs: Any) -> Any:
            calls.append({"name": name, **kwargs})
            return pd.DataFrame(
                {
                    "Hp": [kwargs.get("Hp_init", 30000.0)],
                    "TAS": [250.0],
                    "M": [0.78],
                    "ROCD": [0.0],
                    "mass": [kwargs.get("m_init", 60000.0)],
                    "LAT": [kwargs.get("Lat", 43.0)],
                    "LON": [kwargs.get("Lon", 2.0)],
                    "HDGTrue": [(kwargs.get("initialHeading") or {}).get("true", 90.0)],
                }
            )

        return stub

    return calls, make_stub


def _patch_tcl(
    monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], make_stub: Any
) -> None:
    for fn_name in (
        "constantSpeedLevel",
        "constantSpeedROCD_time",
        "constantSpeedRating_time",
        "accDec_time",
    ):
        monkeypatch.setattr(predictor_mod, fn_name, make_stub(fn_name))
    monkeypatch.setattr(predictor_mod, "_HAS_PYBADA", True)


def test_run_bada_step_passes_lateral_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, make_stub = _make_recorder()
    _patch_tcl(monkeypatch, calls, make_stub)

    _run_bada_step(
        ac=_FakeAC(),
        speed_type="CAS",
        v_init=250.0,
        v_target=250.0,
        phase="Cruise",
        hp_init=30000.0,
        m_init=60000.0,
        delta_temp=0.0,
        config="CR",
        ws=0.0,
        rocd_target=0.0,
        speed_diff_ratio=0.0,
        current_lat=43.0,
        current_lon=2.0,
        current_heading_deg=90.0,
    )

    assert len(calls) == 1
    kw = calls[0]
    assert kw["Lat"] == 43.0
    assert kw["Lon"] == 2.0
    assert kw["initialHeading"] == {"true": 90.0, "constantHeading": True, "magnetic": None}


def _make_flight_parquet(tmp_path: Path, *, with_lateral: bool = True) -> Path:
    n = 2
    cols: dict[str, Any] = {
        "alt_std_m": [10000.0, 10000.0],
        "tas_ms": [250.0, 250.0],
        "temperature": [223.0, 223.0],
        "mach": [0.78, 0.78],
        "cas_ms": [150.0, 150.0],
        "cas_sel_ms": [150.0, 150.0],
        "mach_sel": [0.0, 0.0],
        "vz_sel_ms": [0.0, 0.0],
        "alt_sel_m": [10000.0, 10000.0],
        "long_wind_ms": [0.0, 0.0],
    }
    if with_lateral:
        cols["raw_lat_deg"] = [43.0, 43.0]
        cols["raw_lon_deg"] = [2.0, 2.0]
        cols["fdm_heading_rad"] = [math.pi / 2, math.pi / 2]
        cols["fdm_heading_target_rad"] = [math.pi / 2, math.pi / 2]
        cols["fdm_heading_target_known"] = [True, True]
    df = pl.DataFrame(cols)
    p = tmp_path / "flight.parquet"
    df.write_parquet(p)
    _ = n
    return p


def test_process_single_flight_initializes_lateral_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, make_stub = _make_recorder()
    _patch_tcl(monkeypatch, calls, make_stub)
    p = _make_flight_parquet(tmp_path)

    process_single_flight(p, _FakeAC())

    assert calls, "expected at least one TCL call"
    first = calls[0]
    assert first["Lat"] == pytest.approx(43.0)
    assert first["Lon"] == pytest.approx(2.0)
    assert first["initialHeading"]["true"] == pytest.approx(90.0)


def test_process_single_flight_updates_state_from_step_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    calls: list[dict[str, Any]] = []

    def stub(**kwargs: Any) -> Any:
        calls.append({**kwargs})
        return pd.DataFrame(
            {
                "Hp": [30000.0],
                "TAS": [250.0],
                "M": [0.78],
                "ROCD": [0.0],
                "mass": [60000.0],
                "LAT": [44.0],
                "LON": [3.0],
                "HDGTrue": [95.0],
            }
        )

    for fn_name in (
        "constantSpeedLevel",
        "constantSpeedROCD_time",
        "constantSpeedRating_time",
        "accDec_time",
    ):
        monkeypatch.setattr(predictor_mod, fn_name, stub)
    monkeypatch.setattr(predictor_mod, "_HAS_PYBADA", True)

    p = _make_flight_parquet(tmp_path)
    process_single_flight(p, _FakeAC())

    assert len(calls) >= 2
    second = calls[1]
    assert second["Lat"] == pytest.approx(44.0)
    assert second["Lon"] == pytest.approx(3.0)
    # Heading on second call: target is known so commanded uses target (90°),
    # but the *current* heading state is updated from HDGTrue=95.
    # AC4 says: when target known → use target. AC3 says state updates.
    # initialHeading.true is the commanded heading (target when known).
    assert second["initialHeading"]["true"] == pytest.approx(90.0)


def test_commanded_heading_falls_back_when_target_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    calls: list[dict[str, Any]] = []

    def stub(**kwargs: Any) -> Any:
        calls.append({**kwargs})
        return pd.DataFrame(
            {
                "Hp": [30000.0],
                "TAS": [250.0],
                "M": [0.78],
                "ROCD": [0.0],
                "mass": [60000.0],
                "LAT": [44.0],
                "LON": [3.0],
                "HDGTrue": [95.0],
            }
        )

    for fn_name in (
        "constantSpeedLevel",
        "constantSpeedROCD_time",
        "constantSpeedRating_time",
        "accDec_time",
    ):
        monkeypatch.setattr(predictor_mod, fn_name, stub)
    monkeypatch.setattr(predictor_mod, "_HAS_PYBADA", True)

    df = pl.DataFrame(
        {
            "alt_std_m": [10000.0, 10000.0],
            "tas_ms": [250.0, 250.0],
            "temperature": [223.0, 223.0],
            "mach": [0.78, 0.78],
            "cas_ms": [150.0, 150.0],
            "cas_sel_ms": [150.0, 150.0],
            "mach_sel": [0.0, 0.0],
            "vz_sel_ms": [0.0, 0.0],
            "alt_sel_m": [10000.0, 10000.0],
            "long_wind_ms": [0.0, 0.0],
            "raw_lat_deg": [43.0, 43.0],
            "raw_lon_deg": [2.0, 2.0],
            "fdm_heading_rad": [math.pi / 2, math.pi / 2],
            "fdm_heading_target_rad": [0.0, 0.0],
            "fdm_heading_target_known": [False, False],
        }
    )
    p = tmp_path / "flight.parquet"
    df.write_parquet(p)

    process_single_flight(p, _FakeAC())

    assert len(calls) >= 2
    # First call: initial heading from fdm_heading_rad (90°) since target unknown
    assert calls[0]["initialHeading"]["true"] == pytest.approx(90.0)
    # Second call: target unknown → reuse current heading (updated to 95° from prev step)
    assert calls[1]["initialHeading"]["true"] == pytest.approx(95.0)


def test_output_parquet_has_lateral_columns_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, make_stub = _make_recorder()
    _patch_tcl(monkeypatch, calls, make_stub)
    p = _make_flight_parquet(tmp_path, with_lateral=True)

    out = process_single_flight(p, _FakeAC())

    assert out is not None
    assert "bada_lat_deg" in out.columns
    assert "bada_lon_deg" in out.columns
    assert "bada_heading_rad" in out.columns


def test_output_parquet_omits_lateral_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    def stub(**kwargs: Any) -> Any:
        # Longitudinal-only response (no LAT/LON/HDGTrue cols)
        return pd.DataFrame(
            {
                "Hp": [30000.0],
                "TAS": [250.0],
                "M": [0.78],
                "ROCD": [0.0],
                "mass": [60000.0],
            }
        )

    for fn_name in (
        "constantSpeedLevel",
        "constantSpeedROCD_time",
        "constantSpeedRating_time",
        "accDec_time",
    ):
        monkeypatch.setattr(predictor_mod, fn_name, stub)
    monkeypatch.setattr(predictor_mod, "_HAS_PYBADA", True)

    p = _make_flight_parquet(tmp_path, with_lateral=False)
    out = process_single_flight(p, _FakeAC())

    assert out is not None
    assert "bada_lat_deg" not in out.columns
    assert "bada_lon_deg" not in out.columns
    assert "bada_heading_rad" not in out.columns
    # Longitudinal columns intact
    assert "bada_alt_std_m" in out.columns
    assert "bada_tas_ms" in out.columns


_ = np  # silence unused if path changes
