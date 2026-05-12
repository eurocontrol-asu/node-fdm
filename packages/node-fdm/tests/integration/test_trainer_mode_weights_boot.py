from __future__ import annotations

import polars as pl
import pytest
import structlog
from structlog.testing import capture_logs

from node_fdm.loader import get_train_val_data
from node_fdm.training.weighting import boot_mode_weights

pytestmark = pytest.mark.integration


def _make_training_df() -> pl.DataFrame:
    rows = 200
    labels = ["TURN" if i % 5 == 0 else "ALT_MACH" for i in range(rows)]
    return pl.DataFrame(
        {
            "meta_split": ["train"] * rows,
            "meta_flight_id": [i // 60 for i in range(rows)],
            "fdm_mode_label": labels,
        }
    )


def test_boot_mode_weights_emits_log_and_attaches_column() -> None:
    structlog.reset_defaults()
    df = _make_training_df()
    with capture_logs() as logs:
        out = boot_mode_weights(df)
    events = [e for e in logs if e.get("event") == "mode_weights_computed"]
    assert len(events) == 1
    evt = events[0]
    for key in (
        "alpha",
        "imbalance_ratio",
        "n_labels",
        "top5_labels",
        "bottom5_labels",
        "weight_ratio_max_over_min",
    ):
        assert key in evt
    assert "fdm_train_weight" in out.columns


def _make_loader_df(*, with_weights: bool) -> pl.DataFrame:
    """Build a DF rich enough for ``get_train_val_data`` (1 train + 1 val flight)."""
    rows_per_flight = 80
    rows = rows_per_flight * 2
    base = pl.DataFrame(
        {
            "meta_split": ["train"] * rows_per_flight + ["val"] * rows_per_flight,
            "meta_flight_id": [0] * rows_per_flight + [1] * rows_per_flight,
            "meta_aircraft_type": ["A320"] * rows,
            "fdm_mode_label": ["TURN" if i % 5 == 0 else "ALT_MACH" for i in range(rows)],
            "x0": [float(i) for i in range(rows)],
            "u0": [0.0] * rows,
            "e0": [0.0] * rows,
            "dx0": [0.0] * rows,
            "fdm_flag_valid": [True] * rows,
        }
    )
    if with_weights:
        return boot_mode_weights(base)
    return base


def test_loader_attaches_weight_tensor_when_column_present() -> None:
    """End-to-end: enriching the DF with fdm_train_weight before the loader
    must yield samples whose ``w`` tensor is set. This guards the bug where
    boot ran AFTER dataset construction, leaving every sample with w=None.
    """
    df = _make_loader_df(with_weights=True)
    train_ds, _ = get_train_val_data(
        data_df=df,
        x_cols=["x0"],
        u_cols=["u0"],
        e_cols=["e0"],
        dx_cols=["dx0"],
        seq_len=20,
        shift=20,
    )
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample.w is not None
    assert sample.w.shape == (20,)


def test_loader_yields_none_weight_when_column_missing() -> None:
    """Symmetric guard: when the DF has no fdm_train_weight, samples MUST
    have w=None (the trainer's loss branch on w_tensor is None then kicks
    in correctly)."""
    df = _make_loader_df(with_weights=False)
    train_ds, _ = get_train_val_data(
        data_df=df,
        x_cols=["x0"],
        u_cols=["u0"],
        e_cols=["e0"],
        dx_cols=["dx0"],
        seq_len=20,
        shift=20,
    )
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample.w is None
