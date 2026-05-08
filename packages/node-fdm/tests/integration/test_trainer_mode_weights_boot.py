from __future__ import annotations

import polars as pl
import pytest
import structlog
from structlog.testing import capture_logs

from node_fdm.trainer import TrainingConfig
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


def _maybe_boot(cfg: TrainingConfig, df: pl.DataFrame) -> pl.DataFrame:
    """Mirror the boot gating that ODETrainer.__init__ performs."""
    if cfg.use_mode_weights:
        return boot_mode_weights(df)
    return df


def test_trainer_attaches_weights_when_flag_on() -> None:
    structlog.reset_defaults()
    cfg = TrainingConfig(
        architecture_name="node_adsb_v1", model_name="test", use_mode_weights=True
    )
    df = _make_training_df()
    with capture_logs() as logs:
        out = _maybe_boot(cfg, df)
    events = [e for e in logs if e.get("event") == "mode_weights_computed"]
    assert len(events) == 1
    evt = events[0]
    for key in (
        "beta",
        "imbalance_ratio",
        "n_labels",
        "top5_labels",
        "bottom5_labels",
        "weight_ratio_max_over_min",
    ):
        assert key in evt
    assert "fdm_train_weight" in out.columns


def test_trainer_does_not_attach_weights_when_flag_off() -> None:
    structlog.reset_defaults()
    cfg = TrainingConfig(
        architecture_name="node_adsb_v1", model_name="test", use_mode_weights=False
    )
    df = _make_training_df()
    with capture_logs() as logs:
        out = _maybe_boot(cfg, df)
    assert not any(e.get("event") == "mode_weights_computed" for e in logs)
    assert "fdm_train_weight" not in out.columns
