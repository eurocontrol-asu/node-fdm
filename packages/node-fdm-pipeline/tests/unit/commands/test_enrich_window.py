"""Unit tests for the enrich date window."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "raw_timestamp": [
                datetime(2025, 2, 17, 8),
                datetime(2025, 2, 17, 23, 59),
                datetime(2025, 5, 21, 12),
                datetime(2025, 11, 1, 0, 0),
            ],
            "raw_alt_ft": [1000.0, 2000.0, 3000.0, 4000.0],
        }
    )


def test_window_keeps_one_day() -> None:
    """A one-day window keeps only that day, end_date being exclusive."""
    from node_fdm_pipeline.commands.data import _slice_by_date

    out = _slice_by_date(_frame(), "2025-02-17", "2025-02-18")

    assert out.height == 2
    assert out["raw_alt_ft"].to_list() == [1000.0, 2000.0]


def test_end_date_is_exclusive() -> None:
    """A row exactly on end_date is excluded, matching download and decode."""
    from node_fdm_pipeline.commands.data import _slice_by_date

    out = _slice_by_date(_frame(), "2025-05-21", "2025-11-01")

    assert out["raw_alt_ft"].to_list() == [3000.0]


def test_enrichment_validation_rejects_excessive_nulls() -> None:
    from node_fdm_pipeline.commands.data import validate_enriched_frame

    frame = pl.DataFrame(
        {
            "era_temp_K": [280.0, None],
            "era_u_wind_ms": [1.0, None],
            "era_v_wind_ms": [2.0, None],
            "era_tas_kt": [300.0, None],
            "era_mach": [0.7, None],
            "era_cas_kt": [250.0, None],
        }
    )

    with pytest.raises(RuntimeError, match="exceeds configured threshold"):
        validate_enriched_frame(frame, 0.05)


def test_enrichment_validation_reports_complete_output() -> None:
    from node_fdm_pipeline.commands.data import validate_enriched_frame

    frame = pl.DataFrame(
        {
            "era_temp_K": [280.0],
            "era_u_wind_ms": [1.0],
            "era_v_wind_ms": [2.0],
            "era_tas_kt": [300.0],
            "era_mach": [0.7],
            "era_cas_kt": [250.0],
        }
    )

    outcome = validate_enriched_frame(frame, 0.05)

    assert outcome.rows == 1
    assert outcome.max_null_fraction == 0.0


def test_open_bounds() -> None:
    """An empty bound leaves that side open."""
    from node_fdm_pipeline.commands.data import _slice_by_date

    assert _slice_by_date(_frame(), "2025-05-21", "").height == 2
    assert _slice_by_date(_frame(), "", "2025-05-21").height == 2
    assert _slice_by_date(_frame(), "", "").height == 4


def test_window_outside_the_data_is_empty() -> None:
    """A window matching no row yields an empty frame, not an error.

    enrich() checks for this and returns early rather than handing fastmeteo a
    frame with no timestamps to bound its download with.
    """
    from node_fdm_pipeline.commands.data import _slice_by_date

    assert _slice_by_date(_frame(), "2025-07-01", "2025-07-02").is_empty()
