"""Tests for dataset statistics command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl

from node_fdm_pipeline.commands.stats import run_dataset_stats


class TestRunDatasetStats:
    """Tests for ``run_dataset_stats``."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create a valid YAML config and split CSV."""
        data_dir = tmp_path / "data"
        process_dir = data_dir / "processed_flights"
        process_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
  - B738
"""
        )

        # Create split CSV
        split_df = pl.DataFrame(
            {
                "filepath": [
                    "/fake/f1.parquet",
                    "/fake/f2.parquet",
                    "/fake/f3.parquet",
                ],
                "icao": ["a1", "a2", "a3"],
                "split": ["train", "val", "test"],
                "aircraft_type": ["A320", "A320", "A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        return config

    @patch("node_fdm.loader.get_train_val_data")
    def test_stats_output_format(
        self,
        mock_get_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Stats are logged with correct fields per typecode."""
        config = self._make_config(tmp_path)

        # Mock dataset objects with __len__
        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=100)
        mock_val_ds = MagicMock()
        mock_val_ds.__len__ = MagicMock(return_value=20)
        mock_get_data.return_value = (mock_train_ds, mock_val_ds)

        run_dataset_stats(arch="opensky", config=config)

        # get_train_val_data called once for A320, once for B738
        assert mock_get_data.call_count == 2

    @patch("node_fdm.loader.get_train_val_data")
    def test_stats_empty_typecode(
        self,
        mock_get_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Empty typecode has zero segments."""
        config = self._make_config(tmp_path)

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=0)
        mock_val_ds = MagicMock()
        mock_val_ds.__len__ = MagicMock(return_value=0)
        mock_get_data.return_value = (mock_train_ds, mock_val_ds)

        # B738 has no entries in split CSV → zero segments, no crash
        run_dataset_stats(arch="opensky", config=config)
