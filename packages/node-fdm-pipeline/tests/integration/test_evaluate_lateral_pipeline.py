"""Integration test for the full lateral run of run_evaluate (AXM-1685).

Exercises the real filesystem boundary: writes synthetic Delta + predict +
BADA fixtures into a tmp_path and asserts the resulting performance.parquet
includes Heading and Position rows for both PRED and BADA models.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest
from pytest_mock import MockerFixture


def _make_flight_frame(flight_id: str, *, with_lateral: bool) -> pl.DataFrame:
    n = 30
    base = {
        "meta_flight_id": [flight_id] * n,
        "meta_aircraft_type": ["A320"] * n,
        "meta_split": ["test"] * n,
        "fdm_flag_valid": [True] * n,
        "fdm_flag_crop_start": [0] * n,
        "fdm_flag_crop_end": [n - 1] * n,
        "raw_timestamp": list(range(n)),
        "raw_vz_ms": [0.0] * 10 + [2.0] * 10 + [-2.0] * 10,
        "raw_alt_m": [1000.0 + i * 10 for i in range(n)],
        "era_tas_ms": [220.0 + i * 0.1 for i in range(n)],
        "fdm_gamma_rad": [math.radians(2.0)] * n,
        "fdm_heading_rad": [math.radians(45.0)] * n,
    }
    if with_lateral:
        base["lat_deg"] = [48.85 + i * 1e-4 for i in range(n)]
        base["lon_deg"] = [2.35 + i * 1e-4 for i in range(n)]
    return pl.DataFrame(base)


def _make_pred_frame(truth: pl.DataFrame, *, with_lateral: bool) -> pl.DataFrame:
    cols = {
        "pred_raw_alt_m": truth["raw_alt_m"] + 5.0,
        "pred_era_tas_ms": truth["era_tas_ms"] + 0.5,
        "pred_fdm_gamma_rad": truth["fdm_gamma_rad"] + math.radians(0.1),
        "pred_fdm_heading_rad": truth["fdm_heading_rad"] + math.radians(0.5),
    }
    if with_lateral:
        cols["pred_lat_deg"] = truth["lat_deg"] + 1e-5
        cols["pred_lon_deg"] = truth["lon_deg"] + 1e-5
    return pl.DataFrame(cols)


def _make_bada_frame(truth: pl.DataFrame, *, with_lateral: bool) -> pl.DataFrame:
    cols = {
        "bada_raw_alt_m": truth["raw_alt_m"] + 8.0,
        "bada_era_tas_ms": truth["era_tas_ms"] + 1.0,
        "bada_fdm_gamma_rad": truth["fdm_gamma_rad"] + math.radians(0.2),
        "bada_fdm_heading_rad": truth["fdm_heading_rad"] + math.radians(1.0),
    }
    if with_lateral:
        cols["bada_lat_deg"] = truth["lat_deg"] + 2e-5
        cols["bada_lon_deg"] = truth["lon_deg"] + 2e-5
    return pl.DataFrame(cols)


@pytest.mark.integration
def test_evaluate_full_lateral_run_produces_complete_table(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """AC2, AC4, AC5, AC6: heading + position rows for PRED and BADA."""
    from node_fdm_pipeline.commands import evaluate as evaluate_mod

    truth = _make_flight_frame("F1", with_lateral=True)
    pred_df = _make_pred_frame(truth, with_lateral=True)
    bada_df = _make_bada_frame(truth, with_lateral=True)

    arch_name = "adsb"
    typecode = "A320"
    sub = f"{arch_name}_{typecode}"

    predict_dir = tmp_path / "predicted"
    bada_dir = tmp_path / "bada"
    data_dir = tmp_path / "data"
    (predict_dir / sub / typecode).mkdir(parents=True)
    (bada_dir / typecode).mkdir(parents=True)
    data_dir.mkdir()

    pred_df.write_parquet(predict_dir / sub / typecode / "F1.parquet")
    bada_df.write_parquet(bada_dir / typecode / "F1.parquet")

    paths = mocker.MagicMock()
    paths.resolve.side_effect = lambda key: {
        "delta_table": tmp_path / "delta",
        "predicted_dir": predict_dir,
        "bada_dir": bada_dir,
    }[key]
    paths.data_dir = data_dir

    cfg = mocker.MagicMock()
    cfg.paths = paths
    cfg.typecodes = [typecode]

    import node_fdm_pipeline.config as config_mod

    mocker.patch.object(config_mod.PipelineConfig, "from_yaml", return_value=cfg)

    info = mocker.MagicMock()
    info.name = arch_name
    import node_fdm_pipeline.resolver as resolver_mod

    mocker.patch.object(resolver_mod, "resolve_architecture", return_value=info)

    import node_fdm_data.delta as delta_mod

    mocker.patch.object(delta_mod, "read_delta_table", return_value=truth)

    evaluate_mod.run_evaluate(arch=arch_name, config=tmp_path / "cfg.yaml")

    output = data_dir / "model_performance" / arch_name / "performance.parquet"
    assert output.exists(), "performance.parquet not written"
    perf = pl.read_parquet(output)

    variables = set(perf["Variable"].unique().to_list())
    assert "Heading [deg]" in variables
    assert "Position [m]" in variables
    assert "Altitude [m]" in variables  # existing variables still present

    models_for_heading = set(
        perf.filter(pl.col("Variable") == "Heading [deg]")["Model"].unique().to_list()
    )
    assert {"PRED", "BADA"} <= models_for_heading

    models_for_position = set(
        perf.filter(pl.col("Variable") == "Position [m]")["Model"].unique().to_list()
    )
    assert {"PRED", "BADA"} <= models_for_position
