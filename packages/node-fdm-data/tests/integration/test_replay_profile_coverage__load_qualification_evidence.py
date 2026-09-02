"""Replay de la qualification depuis les preuves empaquetées."""

from __future__ import annotations

import socket

import polars as pl
import pytest

import node_fdm_data as node_fdm_data_module

pytestmark = pytest.mark.integration

_PROFILE_ID = "opensky26-exp03-v1"
_VERTICAL_CHANNELS = ("alt", "gamma", "vz")


def test_cold_replay_reproduces_frozen_campaign_coverages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC1: le replay local reproduit les deux couvertures figées d'exp03."""

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("qualification replay attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    coverage = node_fdm_data_module.replay_profile_coverage(_PROFILE_ID)

    assert isinstance(coverage, node_fdm_data_module.QualificationCoverage)
    assert coverage.profile_id == _PROFILE_ID
    assert coverage.vertical_pct == 73.9448
    assert coverage.speed_pct == 85.2183


def test_flipping_one_valid_vertical_row_moves_vertical_coverage() -> None:
    """AC3: inverser un seul drapeau vertical valide modifie la couverture."""
    evidence = node_fdm_data_module.load_qualification_evidence(_PROFILE_ID)
    indexed = evidence.retained.with_row_index("_row")
    selected_rows = indexed.filter(
        pl.col("admissible") & pl.col("channel").is_in(_VERTICAL_CHANNELS)
    )
    row_to_flip = selected_rows["_row"][0]
    mutated = indexed.with_columns(
        pl.when(pl.col("_row") == row_to_flip)
        .then(False)
        .otherwise(pl.col("admissible"))
        .alias("admissible")
    ).drop("_row")
    profile = node_fdm_data_module.load_profile(_PROFILE_ID)

    coverage = node_fdm_data_module.replay_coverage_from_frame(mutated, profile)

    assert abs(coverage.vertical_pct - 73.9448) >= 1e-4
