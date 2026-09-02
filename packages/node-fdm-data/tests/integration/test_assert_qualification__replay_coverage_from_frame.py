from __future__ import annotations

import polars as pl
import pytest

from node_fdm_data import load_profile, qualification


def test_assert_coverage_rejects_one_flipped_vertical_row() -> None:
    """AC2: flipping one valid vertical selected flag fails closed."""
    profile_id = "opensky26-exp03-v1"
    evidence = qualification.load_qualification_evidence(profile_id)
    profile = load_profile(profile_id)
    indexed = evidence.retained.with_row_index("_row_index")
    valid_vertical = indexed.filter(
        pl.col("channel").is_in(profile.vertical_cascade) & pl.col("admissible").fill_null(False)
    )
    flipped_index = valid_vertical.row(0, named=True)["_row_index"]
    mutated = indexed.with_columns(
        pl.when(pl.col("_row_index") == flipped_index)
        .then(False)
        .otherwise(pl.col("admissible"))
        .alias("admissible")
    ).drop("_row_index")
    changed_rows = mutated.select(
        (pl.col("admissible") != evidence.retained["admissible"]).sum()
    ).item()
    coverage = qualification.replay_coverage_from_frame(mutated, profile)
    targets = qualification.FrozenTargets(
        vertical_pct=73.9448,
        speed_pct=85.2183,
    )

    assert changed_rows == 1
    with pytest.raises(qualification.QualificationMismatchError) as caught:
        qualification.assert_coverage(coverage, targets)

    message = str(caught.value)
    assert "vertical" in message
    assert "73.9448" in message
    assert str(coverage.vertical_pct) in message


def test_assert_qualification_returns_verified_metric_targets() -> None:
    """AC3: qualification returns targets parsed from verified metrics."""
    targets = qualification.assert_qualification("opensky26-exp03-v1")

    assert targets.vertical_pct == 73.9448
    assert targets.speed_pct == 85.2183
