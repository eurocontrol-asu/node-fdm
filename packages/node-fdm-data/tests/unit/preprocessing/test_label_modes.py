from __future__ import annotations

from typing import Any

import polars as pl

from node_fdm_data.preprocessing.label_modes import (
    classify_speed_regime,
    label_modes,
)

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


def _make_df(n: int, **overrides: Any) -> pl.DataFrame:
    base: dict[str, list[Any]] = {
        "meta_flight_id": ["F1"] * n,
        "fdm_in_turn": [False] * n,
        "fdm_alt_sel_ft": [NAN] * n,
        "fdm_vz_sel_ftmin": [NAN] * n,
        "fdm_gamma_sel_rad": [NAN] * n,
        "fdm_mach_sel": [NAN] * n,
        "fdm_cas_sel_kt": [NAN] * n,
    }
    base.update(overrides)
    schema = {
        "meta_flight_id": pl.Utf8,
        "fdm_in_turn": pl.Boolean,
        "fdm_alt_sel_ft": pl.Float64,
        "fdm_vz_sel_ftmin": pl.Float64,
        "fdm_gamma_sel_rad": pl.Float64,
        "fdm_mach_sel": pl.Float64,
        "fdm_cas_sel_kt": pl.Float64,
    }
    return pl.DataFrame(base, schema=schema)


def test_turn_label_overrides_other_axes() -> None:
    df = _make_df(
        5,
        fdm_in_turn=[True] * 5,
        fdm_alt_sel_ft=[10000.0] * 5,
        fdm_mach_sel=[0.78] * 5,
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["TURN"] * 5


def test_alt_mach_label() -> None:
    df = _make_df(
        4,
        fdm_alt_sel_ft=[10000.0] * 4,
        fdm_mach_sel=[0.78] * 4,
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["ALT_MACH"] * 4


def test_alt_cas_label() -> None:
    df = _make_df(
        4,
        fdm_alt_sel_ft=[10000.0] * 4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0] * 4,
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["ALT_CAS"] * 4


def test_alt_unk_label() -> None:
    df = _make_df(
        4,
        fdm_alt_sel_ft=[10000.0] * 4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["ALT_UNK"] * 4


def test_vz_mach_label() -> None:
    df = _make_df(
        4,
        fdm_vz_sel_ftmin=[-1500.0] * 4,
        fdm_mach_sel=[0.78] * 4,
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["VZ_MACH"] * 4


def test_vz_cas_label() -> None:
    df = _make_df(
        4,
        fdm_vz_sel_ftmin=[-1500.0] * 4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0] * 4,
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["VZ_CAS"] * 4


def test_vz_unk_label() -> None:
    df = _make_df(
        4,
        fdm_vz_sel_ftmin=[-1500.0] * 4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["VZ_UNK"] * 4


def test_gamma_mach_label() -> None:
    df = _make_df(
        4,
        fdm_gamma_sel_rad=[0.05] * 4,
        fdm_mach_sel=[0.78] * 4,
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["GAMMA_MACH"] * 4


def test_gamma_cas_label() -> None:
    df = _make_df(
        4,
        fdm_gamma_sel_rad=[0.05] * 4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0] * 4,
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["GAMMA_CAS"] * 4


def test_gamma_unk_label() -> None:
    df = _make_df(
        4,
        fdm_gamma_sel_rad=[0.05] * 4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["GAMMA_UNK"] * 4


def test_unkvert_mach_label() -> None:
    df = _make_df(
        4,
        fdm_mach_sel=[0.78] * 4,
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["UNKVERT_MACH"] * 4


def test_unkvert_cas_label() -> None:
    df = _make_df(
        4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0] * 4,
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["UNKVERT_CAS"] * 4


def test_unkvert_unk_label() -> None:
    df = _make_df(
        4,
        fdm_mach_sel=[0.70, 0.72, 0.74, 0.76],
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    out = label_modes(df)
    assert out["fdm_mode_label"].to_list() == ["UNKVERT_UNK"] * 4


def test_mach_priority_over_cas_when_both_constant() -> None:
    df = _make_df(
        4,
        fdm_alt_sel_ft=[10000.0] * 4,
        fdm_mach_sel=[0.78] * 4,
        fdm_cas_sel_kt=[250.0] * 4,
    )
    out = label_modes(df)
    labels = out["fdm_mode_label"].to_list()
    assert all(label == "ALT_MACH" for label in labels)
    assert not any(label.endswith("_CAS") for label in labels)


def test_classify_speed_regime_returns_bool_series_per_flight() -> None:
    df = pl.DataFrame(
        {
            "meta_flight_id": ["FA", "FA", "FA", "FB", "FB", "FB"],
            "fdm_mach_sel": [0.78, 0.78, 0.78, 0.70, 0.72, 0.74],
        },
        schema={"meta_flight_id": pl.Utf8, "fdm_mach_sel": pl.Float64},
    )
    mask = classify_speed_regime(df, "fdm_mach_sel")
    assert mask.dtype == pl.Boolean
    assert mask.to_list() == [True, True, True, False, False, False]


def test_label_modes_preserves_row_count_and_column_dtype() -> None:
    n = 50
    df = _make_df(
        n,
        fdm_in_turn=[i % 7 == 0 for i in range(n)],
        fdm_alt_sel_ft=[10000.0 if i < 20 else NAN for i in range(n)],
        fdm_vz_sel_ftmin=[NAN if i < 20 else (-1500.0 if i < 35 else NAN) for i in range(n)],
        fdm_gamma_sel_rad=[NAN if i < 35 else (0.05 if i < 45 else NAN) for i in range(n)],
        fdm_mach_sel=[0.78 if i % 2 == 0 else 0.70 + 0.001 * i for i in range(n)],
        fdm_cas_sel_kt=[250.0 if i % 3 == 0 else 240.0 + 0.5 * i for i in range(n)],
    )
    out = label_modes(df)
    assert out.height == n
    assert out.schema["fdm_mode_label"] == pl.Utf8
    values = set(out["fdm_mode_label"].to_list())
    assert values <= VALID_LABELS


def test_label_modes_idempotent_when_column_already_present() -> None:
    df = _make_df(
        4,
        fdm_alt_sel_ft=[10000.0] * 4,
        fdm_mach_sel=[0.78] * 4,
        fdm_cas_sel_kt=[250.0, 252.0, 254.0, 256.0],
    )
    fresh = label_modes(df)
    seeded = df.with_columns(pl.Series("fdm_mode_label", ["STALE"] * 4, dtype=pl.Utf8))
    second = label_modes(seeded)
    assert "fdm_mode_label_2" not in second.columns
    label_cols = [c for c in second.columns if c == "fdm_mode_label"]
    assert len(label_cols) == 1
    assert second["fdm_mode_label"].to_list() == fresh["fdm_mode_label"].to_list()
