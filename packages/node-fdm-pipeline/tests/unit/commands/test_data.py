"""Unit tests for download cache helpers in commands.data.

These tests exercise the in-memory contract of ``_ensure_window_cached``
without real OpenSky I/O: ``_raw_cache.cache_misses`` is patched to control
the miss set and the per-kind fetcher is patched to observe calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from _config_fixtures import selected_param_config

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from node_fdm_pipeline.config import PipelineConfig


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    from node_fdm_pipeline.config import PipelineConfig

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""\
paths:
  data_dir: "{tmp_path / "data"}"

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


def test_ensure_window_cached_skips_when_all_cached(
    cfg: PipelineConfig, mocker: MockerFixture
) -> None:
    """AC1: full cache hit -> no fetch call."""
    from node_fdm_pipeline.commands import data as data_mod

    mocker.patch("node_fdm_pipeline.commands._raw_cache.cache_misses", return_value=[])
    fetch = mocker.patch.object(data_mod, "_fetch_and_cache_window")

    data_mod._ensure_window_cached(cfg, date_str="20240101", icao24_list=["abc123", "def456"])

    assert fetch.call_count == 0


def test_ensure_window_cached_force_ignores_cache(
    cfg: PipelineConfig, mocker: MockerFixture
) -> None:
    """AC3: force=True -> fetch every icao24 regardless of cache."""
    from node_fdm_pipeline.commands import data as data_mod

    cm = mocker.patch("node_fdm_pipeline.commands._raw_cache.cache_misses", return_value=[])
    fetch = mocker.patch.object(data_mod, "_fetch_and_cache_window")

    icao = ["abc123", "def456"]
    data_mod._ensure_window_cached(cfg, date_str="20240101", icao24_list=icao, force=True)

    assert cm.call_count == 0
    assert fetch.call_count >= 1
    for call in fetch.call_args_list:
        kwargs = call.kwargs
        passed = kwargs.get("icao24_misses")
        if passed is None and len(call.args) > 2:
            passed = call.args[2]
        assert passed is not None
        assert list(passed) == icao


def _fake_history_flight() -> Any:
    """Build a minimal ADS-B-only Flight (no BDS columns) for decoder tests."""
    import pandas as pd
    from traffic.core import Flight

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-09-01", periods=3, freq="s", tz="UTC"),
            "icao24": ["abc123"] * 3,
            "callsign": ["TEST123"] * 3,
            "latitude": [48.85, 48.86, 48.87],
            "longitude": [2.35, 2.36, 2.37],
            "altitude": [30000.0, 30100.0, 30200.0],
            "groundspeed": [450.0, 451.0, 452.0],
            "track": [90.0, 91.0, 92.0],
            "vertical_rate": [0.0, 100.0, -50.0],
        }
    )
    return Flight(df)


def test_raw_ehs_decoder_query_ehs_exception_yields_full_bds_schema(
    mocker: MockerFixture,
) -> None:
    """When `query_ehs` raises, decoder still emits every `_BDS_SOURCE_KEYS` column."""
    from node_fdm_pipeline.commands import data as data_mod

    flight = _fake_history_flight()
    mocker.patch.object(type(flight), "query_ehs", side_effect=RuntimeError("rs1090 boom"))

    decoder = data_mod._RawEHSDecoder(rawdata=None)
    out = decoder(flight)

    assert out is not None
    cols = set(out.data.columns)
    for key in data_mod._BDS_SOURCE_KEYS:
        assert key in cols, f"missing BDS source key: {key}"


def test_raw_ehs_decoder_missing_bds_subframe_yields_full_bds_schema(
    mocker: MockerFixture,
) -> None:
    """When decoded.data lacks bds40/50/60, decoder still emits every BDS source key."""
    import pandas as pd

    from node_fdm_pipeline.commands import data as data_mod

    flight = _fake_history_flight()
    fake_decoded = mocker.Mock()
    fake_decoded.data = pd.DataFrame({"timestamp": flight.data["timestamp"]})
    mocker.patch.object(type(flight), "query_ehs", return_value=fake_decoded)

    decoder = data_mod._RawEHSDecoder(rawdata=None)
    out = decoder(flight)

    assert out is not None
    cols = set(out.data.columns)
    for key in data_mod._BDS_SOURCE_KEYS:
        assert key in cols, f"missing BDS source key: {key}"


def test_flight_with_empty_bds_keys_preserves_existing_columns() -> None:
    """The helper must not overwrite columns the flight already carries."""
    import pandas as pd
    from traffic.core import Flight

    from node_fdm_pipeline.commands import data as data_mod

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-09-01", periods=2, freq="s", tz="UTC"),
            "icao24": ["abc123", "abc123"],
            "TAS": [450.0, 451.0],
        }
    )
    out = data_mod._flight_with_empty_bds_keys(Flight(df))
    assert list(out.data["TAS"]) == [450.0, 451.0]
    for key in data_mod._BDS_SOURCE_KEYS:
        assert key in out.data.columns


def _make_synthetic_flight(invalid_idx: list[int]) -> tuple[Any, Any]:
    """Build a 3-phase climb/cruise/descent synthetic flight (300 rows).

    Returns ``(flight_df, valid_mask)``. Cruise at Mach 0.78 (idx 100..199);
    climb/descent at CAS 280/270 kt.
    """
    import numpy as np
    import polars as pl
    from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas

    n_climb, n_cruise, n_descent = 100, 100, 100
    n = n_climb + n_cruise + n_descent
    ft_to_m = 0.3048
    kt_to_ms = 0.514444
    ms_to_kt = 1.0 / kt_to_ms
    alt_ft = np.empty(n, dtype=np.float64)
    alt_ft[:n_climb] = np.linspace(5_000.0, 35_000.0, n_climb)
    alt_ft[n_climb : n_climb + n_cruise] = 35_000.0
    alt_ft[n_climb + n_cruise :] = np.linspace(35_000.0, 5_000.0, n_descent)
    alt_m = alt_ft * ft_to_m
    mach = np.full(n, np.nan)
    mach[n_climb : n_climb + n_cruise] = 0.78
    cas_kt = np.full(n, np.nan)
    cas_kt[:n_climb] = 280.0
    cas_kt[n_climb + n_cruise :] = 270.0
    tas_ms = np.full(n, np.nan)
    tas_ms[:n_climb] = np.asarray(cas_to_tas(cas_kt[:n_climb] * kt_to_ms, alt_m[:n_climb]))
    tas_ms[n_climb : n_climb + n_cruise] = np.asarray(
        mach_to_tas(0.78, alt_m[n_climb : n_climb + n_cruise])
    )
    tas_ms[n_climb + n_cruise :] = np.asarray(
        cas_to_tas(cas_kt[n_climb + n_cruise :] * kt_to_ms, alt_m[n_climb + n_cruise :])
    )
    vz_ftmin = np.zeros(n, dtype=np.float64)
    vz_ftmin[:n_climb] = 1500.0
    vz_ftmin[n_climb + n_cruise :] = -1500.0
    valid = np.ones(n, dtype=bool)
    valid[np.array(invalid_idx)] = False
    flight_df = pl.DataFrame(
        {
            "raw_timestamp": np.arange(n, dtype=np.int64) * 4,
            "raw_alt_ft": alt_ft,
            "raw_vz_ftmin": vz_ftmin,
            "bds_mach_clean": mach,
            "bds_ias_kt_clean": cas_kt,
            "fdm_tas_from_cas_kt": tas_ms * ms_to_kt,
            "fdm_flag_valid": valid,
        }
    )
    return flight_df, valid


def test_build_selected_params_with_valid_filter_preserves_row_count() -> None:
    """Fix A: row count is preserved and produced columns are NaN on invalid rows."""
    import numpy as np

    from node_fdm_pipeline.commands.data import _build_selected_params_with_valid_filter

    flight_df, valid = _make_synthetic_flight([5, 12, 25, 60, 130, 145, 170, 240, 260, 295])
    out = _build_selected_params_with_valid_filter(flight_df, selected_param_config().model_dump())

    assert len(out) == len(flight_df)
    invalid_pos = ~valid
    expected_produced = ("fdm_alt_sel_ft", "fdm_mach_sel", "fdm_cas_sel_kt", "fdm_tas_sel_kt")
    for col in expected_produced:
        assert col in out.columns, f"missing produced column {col}"
        arr = out[col].to_numpy()
        finite_on_invalid = int(np.isfinite(arr[invalid_pos]).sum())
        assert finite_on_invalid == 0, f"{col}: {finite_on_invalid} finite values on invalid rows"

    # Detector ran on the cruise: at least one valid cruise row has a Mach
    # plateau value, confirming detection actually fired.
    mach_sel = out["fdm_mach_sel"].to_numpy()
    cruise_valid = np.zeros(len(flight_df), dtype=bool)
    cruise_valid[100:200] = True
    cruise_valid &= valid
    assert int(np.isfinite(mach_sel[cruise_valid]).sum()) > 0


def test_build_selected_params_with_valid_filter_idempotent() -> None:
    """Fix A: re-running the helper on its own output yields the same values
    (overwrite path triggers when produced columns already exist)."""
    import numpy as np

    from node_fdm_pipeline.commands.data import _build_selected_params_with_valid_filter

    flight_df, _ = _make_synthetic_flight([5, 12, 25, 60, 130, 145, 170, 240, 260, 295])
    cfg = selected_param_config().model_dump()
    out1 = _build_selected_params_with_valid_filter(flight_df, cfg)
    out2 = _build_selected_params_with_valid_filter(out1, cfg)
    for col in ("fdm_alt_sel_ft", "fdm_mach_sel", "fdm_cas_sel_kt", "fdm_tas_sel_kt"):
        a = out1[col].to_numpy()
        b = out2[col].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[finite], b[finite], atol=1e-10), f"{col} not stable across re-runs"
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{col} NaN mask drifted"


def test_build_selected_params_with_valid_filter_no_flag_column_passthrough() -> None:
    """Without ``fdm_flag_valid`` the helper falls back to plain build_selected_params."""
    import numpy as np
    import polars as pl

    from node_fdm_pipeline.commands.data import _build_selected_params_with_valid_filter

    n = 60
    flight_df = pl.DataFrame(
        {
            "raw_timestamp": np.arange(n, dtype=np.int64) * 4,
            "raw_alt_ft": np.linspace(5_000.0, 30_000.0, n),
            "raw_vz_ftmin": np.full(n, 1000.0),
            "bds_mach_clean": np.full(n, np.nan),
            "bds_ias_kt_clean": np.full(n, 250.0),
            "fdm_tas_from_cas_kt": np.full(n, 300.0),
        }
    )
    out = _build_selected_params_with_valid_filter(flight_df, selected_param_config().model_dump())
    assert len(out) == n
    # Detector must have produced at least one new fdm_*_sel* column.
    new_cols = [c for c in out.columns if c.startswith("fdm_") and "_sel" in c]
    assert new_cols, "no fdm_*_sel* columns produced"


def test_build_selected_params_with_valid_filter_forwards_science_profile(
    mocker: MockerFixture,
) -> None:
    """A pinned profile replaces the legacy channel dictionary atomically."""
    import polars as pl

    from node_fdm_pipeline.commands.data import _build_selected_params_with_valid_filter

    flight_df = pl.DataFrame({"raw_timestamp": [0, 4]})
    detector = mocker.patch(
        "node_fdm_data.segments.build_selected_params",
        return_value=flight_df,
    )

    result = _build_selected_params_with_valid_filter(
        flight_df,
        None,
        profile="opensky26-exp03-v1",
    )

    assert result.equals(flight_df)
    detector.assert_called_once_with(
        flight_df,
        None,
        profile="opensky26-exp03-v1",
    )
