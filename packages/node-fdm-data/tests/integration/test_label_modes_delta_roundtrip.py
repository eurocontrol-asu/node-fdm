from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from node_fdm_data.delta import read_delta_table, write_columns
from node_fdm_data.preprocessing.label_modes import label_modes

VALID_LABELS: frozenset[str] = frozenset(
    {
        "TURN",
        "ALT_MACH",
        "ALT_CAS",
        "ALT_UNK",
        "VZ_MACH",
        "VZ_CAS",
        "VZ_UNK",
        "GAMMA_MACH",
        "GAMMA_CAS",
        "GAMMA_UNK",
        "UNKVERT_MACH",
        "UNKVERT_CAS",
        "UNKVERT_UNK",
    }
)

NAN: float = float("nan")


@pytest.mark.integration
def test_label_modes_writes_column_in_place_via_delta(tmp_path: Path) -> None:
    delta_path = tmp_path / "flights.delta"
    seed = pl.DataFrame(
        {
            "meta_flight_id": ["F1"] * 10,
            "fdm_in_turn": [
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "fdm_alt_sel_ft": [10000.0] * 3 + [NAN] * 7,
            "fdm_vz_sel_ftmin": [NAN] * 3 + [-1500.0] * 3 + [NAN] * 4,
            "fdm_gamma_sel_rad": [NAN] * 6 + [0.05, 0.05, 0.05, NAN],
            "fdm_mach_sel": [0.78] * 10,
            "fdm_cas_sel_kt": [
                250.0,
                252.0,
                254.0,
                256.0,
                258.0,
                260.0,
                262.0,
                264.0,
                266.0,
                268.0,
            ],
        },
        schema={
            "meta_flight_id": pl.Utf8,
            "fdm_in_turn": pl.Boolean,
            "fdm_alt_sel_ft": pl.Float64,
            "fdm_vz_sel_ftmin": pl.Float64,
            "fdm_gamma_sel_rad": pl.Float64,
            "fdm_mach_sel": pl.Float64,
            "fdm_cas_sel_kt": pl.Float64,
        },
    )
    seed.write_delta(str(delta_path))

    df = read_delta_table(delta_path)
    labelled = label_modes(df)
    write_columns(labelled, delta_path)

    out = read_delta_table(delta_path)
    assert "fdm_mode_label" in out.columns
    assert out.height == 10
    assert set(out["fdm_mode_label"].to_list()) <= VALID_LABELS
