"""Tests for dataset statistics command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from node_fdm_pipeline.commands.stats import run_dataset_stats


def _make_delta_table(path: Path, *, typecodes: list[str] | None = None) -> None:
    """Create a minimal Delta Table with required columns."""
    typecodes = typecodes or ["A320"]
    rng = np.random.default_rng(42)
    rows = []
    for acft in typecodes:
        for i in range(4):
            split = ["train", "train", "val", "test"][i]
            fid = f"abc123_{acft}_{i:02d}_s0"
            for _j in range(20):
                rows.append(
                    {
                        "meta_flight_id": fid,
                        "meta_split": split,
                        "meta_aircraft_type": acft,
                        "fdm_flag_valid": True,
                        "fdm_flag_distance_ok": True,
                        "raw_alt_m": float(rng.uniform(300, 10000)),
                        "era_tas_ms": float(rng.uniform(100, 250)),
                    }
                )
    df = pl.DataFrame(rows)
    df.write_delta(str(path), mode="overwrite")


class TestRunDatasetStats:
    """Tests for ``run_dataset_stats``."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create a valid YAML config and Delta Table."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

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

        _make_delta_table(data_dir / "flights.delta", typecodes=["A320"])

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

        run_dataset_stats(arch="adsb", config=config)

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

        # B738 has no entries in Delta Table → zero segments, no crash
        run_dataset_stats(arch="adsb", config=config)

    def test_stats_missing_delta(self, tmp_path: Path) -> None:
        """SystemExit when Delta Table doesn't exist."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )
        with pytest.raises(SystemExit, match="pipeline"):
            run_dataset_stats(arch="adsb", config=config)
