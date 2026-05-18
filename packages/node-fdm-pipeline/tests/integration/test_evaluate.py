"""Tests for the evaluation command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl


def _make_gt_df(
    *,
    n: int = 50,
    acft: str = "A320",
    flight_id: str = "F001",
) -> pl.DataFrame:
    """Build a ground-truth DataFrame matching the Delta Table schema."""
    return pl.DataFrame(
        {
            "meta_flight_id": [flight_id] * n,
            "meta_aircraft_type": [acft] * n,
            "meta_split": ["test"] * n,
            "fdm_flag_valid": [True] * n,
            "fdm_flag_crop_start": [0] * n,
            "fdm_flag_crop_end": [n - 1] * n,
            "raw_timestamp": np.arange(n, dtype=np.int64),
            "raw_alt_m": np.linspace(1500, 3000, n),
            "era_tas_ms": np.linspace(200, 250, n),
            "fdm_gamma_rad": np.linspace(-0.01, 0.01, n),
            "raw_vz_ms": np.concatenate(
                [np.full(n // 3, 2.0), np.full(n - 2 * (n // 3), 0.0), np.full(n // 3, -2.0)]
            ),
            "raw_alt_ft": np.linspace(5000, 10000, n),
        }
    )


# ---------------------------------------------------------------------------
# Tests for evaluate_typecode (new Delta Table signature)
# ---------------------------------------------------------------------------


class TestEvaluateTypecode:
    """Tests for ``evaluate_typecode``."""

    @staticmethod
    def _make_bada_parquet(path: Path, n: int = 50) -> None:
        """Write a BADA prediction parquet (bada_ prefixed columns only)."""
        df = pl.DataFrame(
            {
                "bada_raw_alt_m": np.linspace(1550, 3050, n),
                "bada_era_tas_ms": np.linspace(201, 251, n),
                "bada_fdm_gamma_rad": np.linspace(-0.009, 0.011, n),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    @staticmethod
    def _make_nodfdm_parquet(path: Path, n: int = 50) -> None:
        """Write a Node-FDM prediction parquet (pred_ prefixed columns only)."""
        df = pl.DataFrame(
            {
                "pred_raw_alt_m": np.linspace(1510, 3010, n),
                "pred_era_tas_ms": np.linspace(200.5, 250.5, n),
                "pred_fdm_gamma_rad": np.linspace(-0.0095, 0.0105, n),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    def test_skip_no_predictions(self, tmp_path: Path) -> None:
        """Returns empty list when neither bada nor pred dirs exist."""
        from node_fdm_pipeline.commands.evaluate import evaluate_typecode

        acft_df = _make_gt_df()
        result = evaluate_typecode(
            "A320",
            acft_df=acft_df,
            predict_acft_dir=tmp_path / "predict",
            bada_dir=tmp_path / "bada",
            variables={"raw_alt_m": "Altitude"},
        )
        assert result == []

    def test_evaluate_with_predictions(self, tmp_path: Path) -> None:
        """Returns metrics DataFrames for a typecode with predictions."""
        from node_fdm_pipeline.commands.evaluate import evaluate_typecode

        bada_dir = tmp_path / "bada"
        predict_dir = tmp_path / "predict"

        acft_df = _make_gt_df(flight_id="F001")
        self._make_bada_parquet(bada_dir / "A320" / "F001.parquet")
        self._make_nodfdm_parquet(predict_dir / "A320" / "F001.parquet")

        variables = {"raw_alt_m": "Altitude [m]", "era_tas_ms": "TAS [m/s]"}
        result = evaluate_typecode(
            "A320",
            acft_df=acft_df,
            predict_acft_dir=predict_dir / "A320",
            bada_dir=bada_dir,
            variables=variables,
        )

        assert len(result) > 0
        for df in result:
            assert "Aircraft" in df.columns
            assert "Variable" in df.columns
            assert "Model" in df.columns

    def test_evaluate_file_error_continues(self, tmp_path: Path) -> None:
        """Evaluation continues when a single flight file fails."""
        from node_fdm_pipeline.commands.evaluate import evaluate_typecode

        bada_dir = tmp_path / "bada"
        predict_dir = tmp_path / "predict"

        # Two flights: F001 valid, F002 has bad prediction file
        n = 50
        acft_df = pl.concat(
            [_make_gt_df(flight_id="F001", n=n), _make_gt_df(flight_id="F002", n=n)]
        )

        self._make_bada_parquet(bada_dir / "A320" / "F001.parquet")

        bad_path = bada_dir / "A320" / "F002.parquet"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_bytes(b"not a parquet")

        variables = {"raw_alt_m": "Altitude [m]"}
        result = evaluate_typecode(
            "A320",
            acft_df=acft_df,
            predict_acft_dir=predict_dir,
            bada_dir=bada_dir,
            variables=variables,
        )

        # Should get metrics from the valid flight
        assert len(result) > 0

    def test_skip_no_parquet_files(self, tmp_path: Path) -> None:
        """Returns empty list when prediction dir exists but has no parquets."""
        from node_fdm_pipeline.commands.evaluate import evaluate_typecode

        bada_dir = tmp_path / "bada"
        (bada_dir / "A320").mkdir(parents=True)
        # Dir exists, but no .parquet files

        acft_df = _make_gt_df()
        result = evaluate_typecode(
            "A320",
            acft_df=acft_df,
            predict_acft_dir=tmp_path / "predict",
            bada_dir=bada_dir,
            variables={"raw_alt_m": "Altitude"},
        )
        assert result == []


# ---------------------------------------------------------------------------
# Tests for run_evaluate
# ---------------------------------------------------------------------------


class TestRunEvaluate:
    """Tests for ``run_evaluate``."""

    def _make_config(self, tmp_path: Path, data_dir: Path) -> Path:
        """Write a minimal pipeline config."""
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )
        return config

    @patch("node_fdm_data.delta.read_delta_table")
    def test_run_evaluate_writes_output(
        self,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full run produces a performance.parquet file."""
        from node_fdm_pipeline.commands.evaluate import run_evaluate

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        bada_dir = data_dir / "bada_flights"
        predict_dir = data_dir / "predicted_flights"

        n = 50
        mock_read_delta.return_value = _make_gt_df(n=n)

        # Create BADA prediction for flight F001
        pred = pl.DataFrame(
            {
                "bada_raw_alt_m": np.linspace(1550, 3050, n),
                "bada_era_tas_ms": np.linspace(201, 251, n),
                "bada_fdm_gamma_rad": np.linspace(-0.009, 0.011, n),
            }
        )
        (bada_dir / "A320").mkdir(parents=True)
        pred.write_parquet(bada_dir / "A320" / "F001.parquet")
        predict_dir.mkdir(parents=True)

        config = self._make_config(tmp_path, data_dir)
        run_evaluate(arch="adsb", config=config)

        output = data_dir / "model_performance" / "node_adsb_v1" / "performance.parquet"
        assert output.exists()
        result = pl.read_parquet(output)
        assert len(result) > 0
        assert "Aircraft" in result.columns
        assert "Phase" in result.columns

    @patch("node_fdm_data.delta.read_delta_table")
    def test_run_evaluate_no_results(
        self,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Logs warning when no predictions exist."""
        from node_fdm_pipeline.commands.evaluate import run_evaluate

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_read_delta.return_value = _make_gt_df()

        config = self._make_config(tmp_path, data_dir)

        # Should not crash, just log warning
        run_evaluate(arch="adsb", config=config)

        # No output file created
        assert not (
            data_dir / "model_performance" / "node_adsb_v1" / "performance.parquet"
        ).exists()
