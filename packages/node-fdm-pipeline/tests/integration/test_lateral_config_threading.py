"""Integration: PipelineConfig.lateral_detection threads down to derive helpers.

Scenario `config-threading`: feeds a known turning flight through
``_augment_lateral_per_flight`` once with default ``LateralDetectionConfig``
and once with an over-the-top ``rate_threshold`` — the override must
produce strictly fewer detected in-turn samples.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from node_fdm_data.preprocessing.derive import _augment_lateral_per_flight

from node_fdm_pipeline.config import LateralDetectionConfig

pytestmark = pytest.mark.integration


def _turning_flight_df() -> pl.DataFrame:
    """Two clear 90-deg turns with straight legs in between."""
    track = np.concatenate(
        [
            np.full(40, 0.0),
            np.linspace(0.0, 90.0, 20),
            np.full(60, 90.0),
            np.linspace(90.0, 180.0, 20),
            np.full(40, 180.0),
        ]
    )
    n = track.size
    lat = np.linspace(45.0, 47.0, n)
    lon = np.linspace(2.0, 5.0, n)
    flight_id = np.zeros(n, dtype=np.int64)
    return pl.DataFrame(
        {
            "flight_id": flight_id,
            "latitude": lat,
            "longitude": lon,
            "track": track,
        }
    )


def test_threshold_override_reduces_detected_turns() -> None:
    """AC2/AC5: raising rate_threshold via config strictly reduces detections."""
    df = _turning_flight_df()

    baseline = _augment_lateral_per_flight(df, lateral_cfg=LateralDetectionConfig().to_params())
    high_threshold = _augment_lateral_per_flight(
        df,
        lateral_cfg=LateralDetectionConfig(rate_threshold=0.50).to_params(),
    )

    n_baseline = int(baseline["fdm_in_turn"].to_numpy().sum())
    n_high = int(high_threshold["fdm_in_turn"].to_numpy().sum())

    assert n_baseline > 0, "fixture must produce baseline detections"
    assert n_high < n_baseline


def test_default_lateral_cfg_matches_no_cfg() -> None:
    """AC4: passing the default LateralDetectionConfig matches lateral_cfg=None."""
    df = _turning_flight_df()
    explicit = _augment_lateral_per_flight(df, lateral_cfg=LateralDetectionConfig().to_params())
    implicit = _augment_lateral_per_flight(df, lateral_cfg=None)
    assert np.array_equal(
        explicit["fdm_in_turn"].to_numpy(),
        implicit["fdm_in_turn"].to_numpy(),
    )
