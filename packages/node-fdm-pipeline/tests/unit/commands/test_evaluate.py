"""Unit tests for node_fdm_pipeline.commands.evaluate (pure, no I/O)."""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_pipeline.commands.evaluate import compute_errors_by_phase


class TestComputeErrorsByPhase:
    """Tests for ``compute_errors_by_phase``."""

    def test_basic_metrics(self) -> None:
        """Known pred/target values → correct Phase, MAE, MAPE, ME columns."""
        n = 100
        df = pl.DataFrame(
            {
                "pred_alt": np.linspace(100, 200, n),
                "alt": np.linspace(100, 200, n) + 1.0,  # constant +1 error
                "vz_ms": np.concatenate(
                    [
                        np.full(30, 2.0),  # climb
                        np.full(40, 0.0),  # level
                        np.full(30, -2.0),  # descent
                    ]
                ),
            }
        )

        result = compute_errors_by_phase(df, pred_col="pred_alt", target_col="alt")

        # Must have Phase column with expected phases
        phases = result["Phase"].to_list()
        assert "All phases" in phases
        assert "Climb" in phases
        assert "Level flight" in phases
        assert "Descent" in phases

        # MAE should be ~1.0 for all phases (constant bias)
        for mae in result["MAE"].to_list():
            assert abs(mae - 1.0) < 0.1

        # ME should be ~-1.0 (pred = target - 1.0)
        for me in result["ME"].to_list():
            assert abs(me - (-1.0)) < 0.1

        # Count column present
        assert "Count" in result.columns

    def test_angle_variable(self) -> None:
        """For gamma (angle) variables, MAPE should be NaN."""
        df = pl.DataFrame(
            {
                "pred_gamma_rad": [0.1, 0.2, 0.3],
                "gamma_rad": [0.15, 0.25, 0.35],
                "vz_ms": [2.0, 0.0, -2.0],
            }
        )

        result = compute_errors_by_phase(df, pred_col="pred_gamma_rad", target_col="gamma_rad")

        # MAPE should be NaN for angle variables
        for mape in result["MAPE (%)"].to_list():
            assert np.isnan(mape)

    def test_empty_phase(self) -> None:
        """Phase with no data points is excluded from results."""
        df = pl.DataFrame(
            {
                "pred_alt": [100.0, 200.0],
                "alt": [105.0, 205.0],
                "vz_ms": [2.0, 2.0],  # all climb, no level/descent
            }
        )

        result = compute_errors_by_phase(df, pred_col="pred_alt", target_col="alt")

        phases = result["Phase"].to_list()
        assert "Climb" in phases
        assert "All phases" in phases
        assert "Level flight" not in phases
        assert "Descent" not in phases

    def test_schema_columns(self) -> None:
        """Result has the expected schema."""
        df = pl.DataFrame(
            {
                "pred_v": [10.0],
                "v": [11.0],
                "vz_ms": [0.0],
            }
        )

        result = compute_errors_by_phase(df, pred_col="pred_v", target_col="v")

        expected_cols = [
            "Phase",
            "MAE",
            "MAE_std",
            "MAPE (%)",
            "MAPE_std",
            "ME",
            "ME_std",
            "Count",
        ]
        assert result.columns == expected_cols
