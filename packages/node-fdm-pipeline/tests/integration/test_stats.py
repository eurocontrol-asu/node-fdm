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

    @pytest.mark.parametrize(
        "train_len, val_len",
        [
            pytest.param(100, 20, id="populated"),
            pytest.param(0, 0, id="empty"),
        ],
    )
    @patch("node_fdm.loader.get_train_val_data")
    def test_stats_runs_for_each_typecode(
        self,
        mock_get_data: MagicMock,
        tmp_path: Path,
        train_len: int,
        val_len: int,
    ) -> None:
        """run_dataset_stats invokes get_train_val_data once per typecode, even when empty."""
        config = self._make_config(tmp_path)

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=train_len)
        mock_val_ds = MagicMock()
        mock_val_ds.__len__ = MagicMock(return_value=val_len)
        mock_get_data.return_value = (mock_train_ds, mock_val_ds)

        run_dataset_stats(arch="adsb", config=config)

        assert mock_get_data.call_count == 2

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
