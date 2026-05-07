from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from node_fdm_bada.predictor import (
    _bank_angle_for_turn,
    _run_bada_step,
    _turn_direction,
    process_single_flight,
)


def _tcl_result(hp: float = 10000.0, tas_kt: float = 486.0, mass: float = 60000.0) -> Any:
    import pandas as pd

    return pd.DataFrame(
        {
            "Hp": [hp, hp],
            "TAS": [tas_kt, tas_kt],
            "M": [0.78, 0.78],
            "ROCD": [0.0, 0.0],
            "mass": [mass, mass],
            "LAT": [45.0, 45.01],
            "LON": [5.0, 5.0],
            "HDGTrue": [90.0, 90.0],
        }
    )


def test_run_bada_step_passes_turn_metrics_when_in_turn(mocker):
    """AC1: explicit turn args are forwarded as turnMetrics to TCL calls."""
    mock_tcl = mocker.patch("node_fdm_bada.predictor.constantSpeedLevel")
    mock_tcl.return_value = _tcl_result()

    _run_bada_step(
        ac=MagicMock(),
        speed_type="CAS",
        v_init=300.0,
        v_target=300.0,
        phase="Cruise",
        hp_init=30000.0,
        m_init=60000.0,
        delta_temp=0.0,
        config="CR",
        ws=0.0,
        rocd_target=0.0,
        speed_diff_ratio=0.0,
        current_lat=45.0,
        current_lon=5.0,
        current_heading_deg=90.0,
        turn_rate_dps=3.0,
        bank_angle_deg=20.0,
        turn_direction="RIGHT",
    )

    kwargs = mock_tcl.call_args.kwargs
    assert kwargs["turnMetrics"] == {
        "rateOfTurn": 3.0,
        "bankAngle": 20.0,
        "directionOfTurn": "RIGHT",
    }


def test_run_bada_step_default_turn_metrics_when_straight(mocker):
    """AC4: with no turn args, turnMetrics keeps its default (rate=0, bank=0, dir=None)."""
    mock_tcl = mocker.patch("node_fdm_bada.predictor.constantSpeedLevel")
    mock_tcl.return_value = _tcl_result()

    _run_bada_step(
        ac=MagicMock(),
        speed_type="CAS",
        v_init=300.0,
        v_target=300.0,
        phase="Cruise",
        hp_init=30000.0,
        m_init=60000.0,
        delta_temp=0.0,
        config="CR",
        ws=0.0,
        rocd_target=0.0,
        speed_diff_ratio=0.0,
        current_lat=45.0,
        current_lon=5.0,
        current_heading_deg=90.0,
    )

    kwargs = mock_tcl.call_args.kwargs
    assert kwargs["turnMetrics"] == {
        "rateOfTurn": 0.0,
        "bankAngle": 0.0,
        "directionOfTurn": None,
    }


def test_bank_angle_derived_from_turn_formula():
    """AC3: bank = degrees(arctan(omega * V / g)) with g=9.80665."""
    omega_rads = 0.05
    tas_ms = 200.0
    g = 9.80665
    expected = math.degrees(math.atan(omega_rads * tas_ms / g))

    result = _bank_angle_for_turn(omega_rads=omega_rads, tas_ms=tas_ms)

    assert result == pytest.approx(expected, abs=1e-6)
    assert result == pytest.approx(45.566, abs=0.1)


def test_turn_direction_from_d_heading_sign():
    """AC2: positive d_heading -> RIGHT, negative -> LEFT (heading clockwise)."""
    assert _turn_direction(0.05) == "RIGHT"
    assert _turn_direction(-0.05) == "LEFT"


def test_in_turn_below_threshold_treated_as_straight(tmp_path, mocker):
    """AC4: |fdm_d_heading_rads| < 1e-3 with fdm_in_turn=True -> default turnMetrics."""
    flight_path = tmp_path / "flight.parquet"
    n = 2
    df = pl.DataFrame(
        {
            "alt_std_m": [10000.0] * n,
            "tas_ms": [250.0] * n,
            "temperature": [223.15] * n,
            "mach_sel": [0.78] * n,
            "alt_sel_m": [10000.0] * n,
            "vz_sel_ms": [0.0] * n,
            "long_wind_ms": [0.0] * n,
            "mach": [0.78] * n,
            "cas_sel_ms": [130.0] * n,
            "raw_lat_deg": [45.0] * n,
            "raw_lon_deg": [5.0] * n,
            "fdm_heading_rad": [math.radians(90.0)] * n,
            "fdm_heading_target_rad": [math.radians(90.0)] * n,
            "fdm_heading_target_known": [True] * n,
            "fdm_in_turn": [True] * n,
            "fdm_d_heading_rads": [1e-4] * n,
        }
    )
    df.write_parquet(flight_path)

    captured: list[dict[str, Any]] = []

    def fake_run(**kwargs):
        captured.append(kwargs)
        return _tcl_result(hp=kwargs["hp_init"], mass=kwargs["m_init"])

    mocker.patch("node_fdm_bada.predictor._HAS_PYBADA", True)
    mocker.patch("node_fdm_bada.predictor._run_bada_step", side_effect=fake_run)

    ac = MagicMock()
    ac.MTOW = 70000.0

    result = process_single_flight(flight_path, ac)

    assert result is not None
    assert len(captured) >= 1
    first = captured[0]
    assert first.get("turn_rate_dps", 0.0) == 0.0
    assert first.get("bank_angle_deg", 0.0) == 0.0
    assert first.get("turn_direction") is None
