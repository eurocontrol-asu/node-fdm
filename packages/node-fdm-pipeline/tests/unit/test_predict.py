from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from node_fdm_pipeline.commands.predict import _predict_flight

LATERAL_X_COLS = ["fdm_heading_rad", "era_tas_ms", "fdm_gamma_rad"]
LATERAL_U_COLS = ["u0"]
LATERAL_E_COLS = ["era_u_wind_ms", "era_v_wind_ms"]

LONG_X_COLS = ["era_tas_ms", "fdm_gamma_rad"]
LONG_U_COLS = ["u0"]
LONG_E_COLS = ["era_temp_k"]


def _make_flight_df(  # noqa: PLR0913
    *,
    n: int,
    x_cols: list[str],
    u_cols: list[str],
    e_cols: list[str],
    lat0: float = 43.0,
    lon0: float = 2.0,
    heading: float = math.pi / 2,
    tas: float = 100.0,
    gamma: float = 0.0,
    u_wind: float = 0.0,
    v_wind: float = 0.0,
) -> pl.DataFrame:
    base = {
        "meta_flight_id": ["F1"] * n,
        "fdm_flag_crop_start": [0] * n,
        "fdm_flag_crop_end": [n - 1] * n,
        "raw_lat_deg": [lat0] * n,
        "raw_lon_deg": [lon0] * n,
        "era_u_wind_ms": [u_wind] * n,
        "era_v_wind_ms": [v_wind] * n,
        "fdm_heading_rad": [heading] * n,
        "era_tas_ms": [tas] * n,
        "fdm_gamma_rad": [gamma] * n,
        "era_temp_k": [288.15] * n,
        "u0": [0.0] * n,
    }
    needed = (
        {
            "meta_flight_id",
            "fdm_flag_crop_start",
            "fdm_flag_crop_end",
            "raw_lat_deg",
            "raw_lon_deg",
            "era_u_wind_ms",
            "era_v_wind_ms",
        }
        | set(x_cols)
        | set(u_cols)
        | set(e_cols)
    )
    return pl.DataFrame({k: v for k, v in base.items() if k in needed})


class _MockPredictor:
    def __init__(self, predictions: dict[str, Any], step: float = 4.0) -> None:
        self._predictions = predictions
        self.meta = SimpleNamespace(step=step)

    def predict_flight(
        self,
        x_init: Any,
        u_seq: Any,
        e_seq: Any,
    ) -> dict[str, Any]:
        return self._predictions


def test_predict_flight_lateral_writes_lat_lon(tmp_path: Path) -> None:
    n = 5
    df = _make_flight_df(n=n, x_cols=LATERAL_X_COLS, u_cols=LATERAL_U_COLS, e_cols=LATERAL_E_COLS)
    info = SimpleNamespace(x_cols=LATERAL_X_COLS, u_cols=LATERAL_U_COLS, e0_cols=LATERAL_E_COLS)
    n_pred = n - 1
    preds = {
        "fdm_heading_rad": np.full(n_pred, math.pi / 2, dtype=np.float32),
        "era_tas_ms": np.full(n_pred, 100.0, dtype=np.float32),
        "fdm_gamma_rad": np.zeros(n_pred, dtype=np.float32),
    }
    predictor = _MockPredictor(preds)

    _predict_flight(
        flight_df=df,
        info=info,
        predictor=predictor,
        output_dir=tmp_path,
        nan_threshold=1.0,
    )
    out = pl.read_parquet(tmp_path / "F1.parquet")
    assert "pred_lat_deg" in out.columns
    assert "pred_lon_deg" in out.columns


def test_predict_flight_longitudinal_no_lat_lon(tmp_path: Path) -> None:
    n = 5
    df = _make_flight_df(n=n, x_cols=LONG_X_COLS, u_cols=LONG_U_COLS, e_cols=LONG_E_COLS)
    info = SimpleNamespace(x_cols=LONG_X_COLS, u_cols=LONG_U_COLS, e0_cols=LONG_E_COLS)
    n_pred = n - 1
    preds = {
        "era_tas_ms": np.full(n_pred, 100.0, dtype=np.float32),
        "fdm_gamma_rad": np.zeros(n_pred, dtype=np.float32),
    }
    predictor = _MockPredictor(preds)

    _predict_flight(
        flight_df=df,
        info=info,
        predictor=predictor,
        output_dir=tmp_path,
        nan_threshold=1.0,
    )
    out = pl.read_parquet(tmp_path / "F1.parquet")
    assert "pred_lat_deg" not in out.columns
    assert "pred_lon_deg" not in out.columns


def test_predict_flight_initial_position_matches_truth(tmp_path: Path) -> None:
    n = 5
    df = _make_flight_df(
        n=n,
        x_cols=LATERAL_X_COLS,
        u_cols=LATERAL_U_COLS,
        e_cols=LATERAL_E_COLS,
        lat0=43.0,
        lon0=2.0,
    )
    info = SimpleNamespace(x_cols=LATERAL_X_COLS, u_cols=LATERAL_U_COLS, e0_cols=LATERAL_E_COLS)
    n_pred = n - 1
    preds = {
        "fdm_heading_rad": np.full(n_pred, math.pi / 2, dtype=np.float32),
        "era_tas_ms": np.full(n_pred, 100.0, dtype=np.float32),
        "fdm_gamma_rad": np.zeros(n_pred, dtype=np.float32),
    }
    predictor = _MockPredictor(preds)

    _predict_flight(
        flight_df=df,
        info=info,
        predictor=predictor,
        output_dir=tmp_path,
        nan_threshold=1.0,
    )
    out = pl.read_parquet(tmp_path / "F1.parquet")
    assert out["pred_lat_deg"][0] == pytest.approx(43.0)
    assert out["pred_lon_deg"][0] == pytest.approx(2.0)


def test_predict_flight_zero_wind_zero_gamma_pure_east(tmp_path: Path) -> None:
    n = 2
    lat0 = 43.0
    lon0 = 2.0
    tas = 100.0
    step = 4.0
    df = _make_flight_df(
        n=n,
        x_cols=LATERAL_X_COLS,
        u_cols=LATERAL_U_COLS,
        e_cols=LATERAL_E_COLS,
        lat0=lat0,
        lon0=lon0,
        heading=math.pi / 2,
        tas=tas,
        gamma=0.0,
        u_wind=0.0,
        v_wind=0.0,
    )
    info = SimpleNamespace(x_cols=LATERAL_X_COLS, u_cols=LATERAL_U_COLS, e0_cols=LATERAL_E_COLS)
    n_pred = n - 1
    preds = {
        "fdm_heading_rad": np.full(n_pred, math.pi / 2, dtype=np.float32),
        "era_tas_ms": np.full(n_pred, tas, dtype=np.float32),
        "fdm_gamma_rad": np.zeros(n_pred, dtype=np.float32),
    }
    predictor = _MockPredictor(preds, step=step)

    _predict_flight(
        flight_df=df,
        info=info,
        predictor=predictor,
        output_dir=tmp_path,
        nan_threshold=1.0,
    )
    out = pl.read_parquet(tmp_path / "F1.parquet")
    earth_radius = 6_371_000.0
    expected_dlon = math.degrees(tas * step / (earth_radius * math.cos(math.radians(lat0))))
    assert out["pred_lon_deg"][1] == pytest.approx(lon0 + expected_dlon, rel=1e-4)
    assert out["pred_lat_deg"][1] == pytest.approx(lat0, abs=1e-6)
