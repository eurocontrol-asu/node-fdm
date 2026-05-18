from __future__ import annotations

import polars as pl
import pytest

from node_fdm.dataset import FlightDataset
from node_fdm.loader import FLIGHT_FEATURE_COLS, get_train_val_data
from node_fdm_data.physics.isa import isa_temperature

X_COLS = ["x"]
U_COLS = ["fdm_sel"]
E_COLS = ["e"]
DX_COLS = ["dx"]


def _flight_rows(
    fid: str,
    split: str,
    *,
    adep: list[float | None],
    ades: list[float | None],
    alt: list[float] | None = None,
    wind: list[float] | None = None,
    temp: list[float] | None = None,
) -> list[dict[str, object]]:
    n_rows = len(adep)
    alt_values = alt if alt is not None else [1000.0 + i * 100.0 for i in range(n_rows)]
    wind_values = wind if wind is not None else [5.0 + i for i in range(n_rows)]
    temp_values = temp if temp is not None else [280.0 + i for i in range(n_rows)]
    return [
        {
            "meta_split": split,
            "meta_flight_id": fid,
            "x": 1.0 + i,
            "fdm_sel": None if i == 0 else float(i),
            "e": 2.0 + i,
            "dx": 0.1 + i,
            "raw_alt_m": alt_values[i],
            "fdm_long_wind_ms": wind_values[i],
            "era_temp_K": temp_values[i],
            "fdm_adep_dist_m": adep[i],
            "fdm_ades_dist_m": ades[i],
        }
        for i in range(n_rows)
    ]


def _dataframe(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def _train_val(
    data_df: pl.DataFrame,
    *,
    flight_feature_cols: list[str] | None = None,
    require_routing: bool = False,
) -> tuple[FlightDataset, FlightDataset]:
    return get_train_val_data(
        data_df,
        X_COLS,
        U_COLS,
        E_COLS,
        DX_COLS,
        seq_len=4,
        shift=1,
        flight_feature_cols=flight_feature_cols,
        require_routing=require_routing,
    )


def test_defaults_unchanged_back_compat() -> None:
    """AC6: default loader calls do not attach flight features."""
    rows = _flight_rows(
        "train-a",
        "train",
        adep=[0.0, 1.0, 2.0, 3.0],
        ades=[4.0, 3.0, 2.0, 1.0],
    ) + _flight_rows(
        "val-a",
        "val",
        adep=[0.0, 1.0, 2.0, 3.0],
        ades=[4.0, 3.0, 2.0, 1.0],
    )

    train_ds, val_ds = _train_val(_dataframe(*rows))

    assert train_ds[0].flight_features is None
    assert val_ds[0].flight_features is None


def test_require_routing_drops_flights_no_features_attached() -> None:
    """AC2: require_routing drops flights with no routing without attaching features."""
    rows = (
        _flight_rows(
            "train-missing",
            "train",
            adep=[None, None, None, None],
            ades=[None, None, None, None],
        )
        + _flight_rows(
            "train-present",
            "train",
            adep=[0.0, 1.0, 2.0, 3.0],
            ades=[4.0, 3.0, 2.0, 1.0],
        )
        + _flight_rows(
            "val-present",
            "val",
            adep=[0.0, 1.0, 2.0, 3.0],
            ades=[4.0, 3.0, 2.0, 1.0],
        )
    )

    train_ds, _ = _train_val(_dataframe(*rows), require_routing=True)

    assert len(train_ds) == 1
    assert train_ds[0].flight_features is None


def test_flight_features_attached_when_requested() -> None:
    """AC1, AC5: requested features are broadcast over each emitted window."""
    rows = _flight_rows(
        "train-a",
        "train",
        adep=[10.0, 20.0, 30.0, 40.0],
        ades=[90.0, 80.0, 70.0, 60.0],
    ) + _flight_rows(
        "val-a",
        "val",
        adep=[1.0, 2.0, 3.0, 4.0],
        ades=[9.0, 8.0, 7.0, 6.0],
    )

    train_ds, _ = _train_val(_dataframe(*rows), flight_feature_cols=FLIGHT_FEATURE_COLS)

    features = train_ds[0].flight_features
    assert features is not None
    assert tuple(features.shape) == (4, 5)
    assert features[:, 0].tolist() == pytest.approx([100.0] * 4, abs=1e-4)
    assert features[:, 1].tolist() == pytest.approx([10.0] * 4, abs=1e-4)


def test_flight_features_auto_promotes_require_routing() -> None:
    """AC2: requesting flight features filters flights without routing."""
    rows = (
        _flight_rows(
            "train-missing",
            "train",
            adep=[None, None, None, None],
            ades=[None, None, None, None],
        )
        + _flight_rows(
            "train-present",
            "train",
            adep=[10.0, 20.0, 30.0, 40.0],
            ades=[90.0, 80.0, 70.0, 60.0],
        )
        + _flight_rows(
            "val-present",
            "val",
            adep=[1.0, 2.0, 3.0, 4.0],
            ades=[9.0, 8.0, 7.0, 6.0],
        )
    )

    train_ds, _ = _train_val(
        _dataframe(*rows),
        flight_feature_cols=FLIGHT_FEATURE_COLS,
        require_routing=False,
    )

    assert len(train_ds) == 1
    assert train_ds[0].flight_features is not None


def test_skips_windows_with_null_t0_distance() -> None:
    """AC3: feature windows with null t0 routing distances are skipped."""
    rows = _flight_rows(
        "train-a",
        "train",
        adep=[None, 10.0, 20.0, 30.0, 40.0],
        ades=[90.0, 80.0, 70.0, 60.0, 50.0],
    ) + _flight_rows(
        "val-a",
        "val",
        adep=[1.0, 2.0, 3.0, 4.0],
        ades=[9.0, 8.0, 7.0, 6.0],
    )

    train_ds, _ = _train_val(_dataframe(*rows), flight_feature_cols=FLIGHT_FEATURE_COLS)

    assert len(train_ds) == 1
    features = train_ds[0].flight_features
    assert features is not None
    assert features[0, 1].item() == pytest.approx(10.0, abs=1e-4)


def test_aggregates_match_hand_computed() -> None:
    """AC4: flight-level aggregates match hand-computed values."""
    alt = [1000.0, 2000.0, 3000.0, 2500.0]
    wind = [1.0, 3.0, 5.0, 7.0]
    temp = [280.0, 281.0, 282.0, 283.0]
    rows = _flight_rows(
        "train-a",
        "train",
        adep=[10.0, 20.0, 30.0, 40.0],
        ades=[90.0, 80.0, 70.0, 60.0],
        alt=alt,
        wind=wind,
        temp=temp,
    ) + _flight_rows(
        "val-a",
        "val",
        adep=[1.0, 2.0, 3.0, 4.0],
        ades=[9.0, 8.0, 7.0, 6.0],
    )
    expected_temp_dev = sum(
        t - float(isa_temperature(a)) for t, a in zip(temp, alt, strict=True)
    ) / len(alt)

    train_ds, _ = _train_val(_dataframe(*rows), flight_feature_cols=FLIGHT_FEATURE_COLS)

    features = train_ds[0].flight_features
    assert features is not None
    assert features[0, 2].item() == pytest.approx(max(alt), abs=1e-3)
    assert features[0, 3].item() == pytest.approx(sum(wind) / len(wind), abs=1e-3)
    assert features[0, 4].item() == pytest.approx(expected_temp_dev, abs=1e-3)


def test_get_train_val_data_threads_kwargs() -> None:
    """AC6, AC7: public loader forwards feature and routing kwargs to both splits."""
    rows = _flight_rows(
        "train-a",
        "train",
        adep=[10.0, 20.0, 30.0, 40.0],
        ades=[90.0, 80.0, 70.0, 60.0],
    ) + _flight_rows(
        "val-a",
        "val",
        adep=[1.0, 2.0, 3.0, 4.0],
        ades=[9.0, 8.0, 7.0, 6.0],
    )
    data_df = _dataframe(*rows)

    train_ds, val_ds = _train_val(data_df, flight_feature_cols=FLIGHT_FEATURE_COLS)
    routing_train_ds, routing_val_ds = _train_val(data_df, require_routing=True)

    train_features = train_ds[0].flight_features
    val_features = val_ds[0].flight_features
    assert train_features is not None
    assert val_features is not None
    assert tuple(train_features.shape) == (4, 5)
    assert tuple(val_features.shape) == (4, 5)
    assert routing_train_ds[0].flight_features is None
    assert routing_val_ds[0].flight_features is None
