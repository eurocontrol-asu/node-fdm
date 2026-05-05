"""Integration test for the lateral channel data pipeline.

Reads the real Delta Table at ``data/flights.delta`` and runs the lateral
columns through :func:`derive_columns` on the 5 reference flights from
``scripts/debug/check_lateral.py``.  Skipped when the Delta is absent so
CI without data still passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from deltalake import DeltaTable

from node_fdm_data.preprocessing.convert import convert_si
from node_fdm_data.preprocessing.derive import derive_columns

pytestmark = pytest.mark.integration


_DEFAULT_FLIGHTS = [
    "738284_ISR826_s0",
    "40666a_EZY78QT_s0",
    "4a0443_BTI7FR_s0",
    "a88774_JBU493_s0",
    "44006e_AUA445_s0",
]

_DELTA_PATH = Path(__file__).resolve().parents[4] / "data" / "flights.delta"


def _load_reference_flights() -> pl.DataFrame:
    if not _DELTA_PATH.exists():
        pytest.skip(f"Delta table not found at {_DELTA_PATH}")
    dt = DeltaTable(str(_DELTA_PATH))
    df = pl.DataFrame(dt.to_pyarrow_table())
    available = set(df["meta_flight_id"].unique().to_list())
    missing = [f for f in _DEFAULT_FLIGHTS if f not in available]
    if missing:
        pytest.skip(f"reference flights not in Delta: {missing}")
    return df.filter(pl.col("meta_flight_id").is_in(_DEFAULT_FLIGHTS))


def test_lateral_columns_present_after_derive() -> None:
    df = _load_reference_flights()
    out = derive_columns(df)

    expected = {
        "fdm_track_clean_deg",
        "fdm_declination_deg",
        "fdm_drift_deg",
        "fdm_wind_std_ms",
        "fdm_heading_deg",
        "fdm_heading_target_deg",
        "fdm_heading_known",
        "fdm_heading_target_known",
        "fdm_in_turn",
        "fdm_track_ortho_deg",
        "fdm_track_sel_known",
    }
    assert expected <= set(out.columns), f"missing: {expected - set(out.columns)}"


def test_lateral_si_conversions() -> None:
    df = _load_reference_flights()
    out = convert_si(derive_columns(df))

    assert "fdm_heading_rad" in out.columns
    assert "fdm_heading_target_rad" in out.columns

    heading_rad = out["fdm_heading_rad"].to_numpy()
    finite = heading_rad[np.isfinite(heading_rad)]
    assert finite.size > 0
    assert (finite >= 0.0).all() and (finite < 2.0 * np.pi + 1e-9).all()

    target_rad = out["fdm_heading_target_rad"].to_numpy()
    finite_t = target_rad[np.isfinite(target_rad)]
    assert finite_t.size > 0
    assert (finite_t >= -np.pi - 1e-9).all() and (finite_t <= np.pi + 1e-9).all()


def test_lateral_heading_coverage() -> None:
    df = _load_reference_flights()
    out = derive_columns(df)
    coverage = float(out["fdm_heading_known"].mean() or 0.0)
    # Briefing observed coverage ~36% sample-weighted over the full Delta;
    # demand a conservative 30% on the 5 reference flights.
    assert coverage > 0.3, f"heading_known coverage too low: {coverage:.2%}"


def test_lateral_target_in_principal_branch() -> None:
    df = _load_reference_flights()
    out = convert_si(derive_columns(df))
    target_rad = out["fdm_heading_target_rad"].to_numpy()
    finite = target_rad[np.isfinite(target_rad)]
    assert finite.size > 0
    # Strict principal-branch check: max(|target|) <= π.
    assert np.abs(finite).max() <= np.pi + 1e-9
