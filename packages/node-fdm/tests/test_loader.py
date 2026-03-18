"""Tests for node_fdm.loader — train/val dataset construction from split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from node_fdm.dataset import FlightDataset
from node_fdm.loader import _load_and_window, get_train_val_data


class TestGetTrainValData:
    """Tests for the public get_train_val_data function."""

    def _make_flights(self, tmp_path: Path, n_rows: int = 100) -> pl.DataFrame:
        """Create parquet files and return a split DataFrame."""
        for split, n_files in [("train", 3), ("val", 1)]:
            for i in range(n_files):
                df = pl.DataFrame(
                    {
                        "alt": np.linspace(1000, 10000, n_rows),
                        "tas": np.linspace(200, 250, n_rows),
                        "cmd": np.random.default_rng(i).random(n_rows).astype(np.float32),
                        "temp": np.full(n_rows, 220.0),
                        "d_alt": np.random.default_rng(i + 100).random(n_rows).astype(np.float32),
                    }
                )
                path = tmp_path / f"{split}_{i:02d}.parquet"
                df.write_parquet(path)

        rows = []
        for split, n_files in [("train", 3), ("val", 1)]:
            for i in range(n_files):
                rows.append(
                    {
                        "filepath": str(tmp_path / f"{split}_{i:02d}.parquet"),
                        "split": split,
                    }
                )
        return pl.DataFrame(rows)

    def test_basic_train_val(self, tmp_path: Path) -> None:
        """Produces non-empty FlightDataset instances."""
        data_df = self._make_flights(tmp_path)
        train_ds, val_ds = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        assert isinstance(train_ds, FlightDataset)
        assert isinstance(val_ds, FlightDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0

    def test_sample_shapes(self, tmp_path: Path) -> None:
        """Each sample has correct tensor shapes."""
        data_df = self._make_flights(tmp_path)
        train_ds, _ = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        sample = train_ds[0]
        assert sample.x.shape == (10, 2)  # seq_len x n_x
        assert sample.u.shape == (10, 1)
        assert sample.e.shape == (10, 1)
        assert sample.dx.shape == (10, 1)

    def test_train_limit(self, tmp_path: Path) -> None:
        """train_limit restricts number of files loaded."""
        data_df = self._make_flights(tmp_path)
        ds_full, _ = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        ds_limited, _ = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
            train_limit=1,
        )
        assert len(ds_limited) < len(ds_full)


class TestLoadAndWindow:
    """Tests for _load_and_window edge cases (bypass FlightDataset min-sample)."""

    def _cols(self) -> tuple[list[str], list[str], list[str], list[str]]:
        return ["alt", "tas"], ["cmd"], ["temp"], ["d_alt"]

    def _make_parquet(self, path: Path, n: int = 30) -> None:
        """Write a valid flight parquet."""
        df = pl.DataFrame(
            {
                "alt": np.linspace(1000, 10000, n),
                "tas": np.linspace(200, 250, n),
                "cmd": np.zeros(n),
                "temp": np.full(n, 220.0),
                "d_alt": np.ones(n),
            }
        )
        df.write_parquet(path)

    def test_missing_columns_skipped(self, tmp_path: Path) -> None:
        """Flight with missing columns produces zero samples."""
        df = pl.DataFrame(
            {
                "alt": np.linspace(1000, 10000, 20),
                "tas": np.linspace(200, 250, 20),
                "cmd": np.zeros(20),
                "temp": np.full(20, 220.0),
                # "d_alt" missing
            }
        )
        path = tmp_path / "bad.parquet"
        df.write_parquet(path)

        x, u, e, dx = self._cols()
        samples = _load_and_window([path], x, u, e, dx, seq_len=10, shift=10)
        assert len(samples) == 0

    def test_too_short_flight(self, tmp_path: Path) -> None:
        """Flight shorter than seq_len produces no samples."""
        df = pl.DataFrame(
            {
                "alt": [1000.0, 2000.0],
                "tas": [200.0, 210.0],
                "cmd": [0.0, 0.1],
                "temp": [220.0, 220.0],
                "d_alt": [1.0, 2.0],
            }
        )
        path = tmp_path / "short.parquet"
        df.write_parquet(path)

        x, u, e, dx = self._cols()
        samples = _load_and_window([path], x, u, e, dx, seq_len=10, shift=10)
        assert len(samples) == 0

    def test_nan_windows_skipped(self, tmp_path: Path) -> None:
        """Windows containing NaN values are excluded."""
        n = 30
        alt = np.linspace(1000, 10000, n)
        alt[5] = np.nan  # NaN in middle

        df = pl.DataFrame(
            {
                "alt": alt,
                "tas": np.linspace(200, 250, n),
                "cmd": np.zeros(n),
                "temp": np.full(n, 220.0),
                "d_alt": np.ones(n),
            }
        )
        path_nan = tmp_path / "nan.parquet"
        df.write_parquet(path_nan)

        # Clean version for comparison
        clean = pl.DataFrame(
            {
                "alt": np.linspace(1000, 10000, n),
                "tas": np.linspace(200, 250, n),
                "cmd": np.zeros(n),
                "temp": np.full(n, 220.0),
                "d_alt": np.ones(n),
            }
        )
        path_clean = tmp_path / "clean.parquet"
        clean.write_parquet(path_clean)

        x, u, e, dx = self._cols()
        nan_samples = _load_and_window([path_nan], x, u, e, dx, seq_len=10, shift=5)
        clean_samples = _load_and_window([path_clean], x, u, e, dx, seq_len=10, shift=5)
        assert len(nan_samples) < len(clean_samples)

    def test_inf_windows_skipped(self, tmp_path: Path) -> None:
        """Windows containing inf values are excluded."""
        n = 30
        alt = np.linspace(1000, 10000, n)
        alt[5] = np.inf  # +inf in first window

        df = pl.DataFrame(
            {
                "alt": alt,
                "tas": np.linspace(200, 250, n),
                "cmd": np.zeros(n),
                "temp": np.full(n, 220.0),
                "d_alt": np.ones(n),
            }
        )
        path_inf = tmp_path / "inf.parquet"
        df.write_parquet(path_inf)

        # Clean version for comparison
        clean = pl.DataFrame(
            {
                "alt": np.linspace(1000, 10000, n),
                "tas": np.linspace(200, 250, n),
                "cmd": np.zeros(n),
                "temp": np.full(n, 220.0),
                "d_alt": np.ones(n),
            }
        )
        path_clean = tmp_path / "clean.parquet"
        clean.write_parquet(path_clean)

        x, u, e, dx = self._cols()
        inf_samples = _load_and_window([path_inf], x, u, e, dx, seq_len=10, shift=5)
        clean_samples = _load_and_window([path_clean], x, u, e, dx, seq_len=10, shift=5)
        assert len(inf_samples) < len(clean_samples)

    def test_neg_inf_windows_skipped(self, tmp_path: Path) -> None:
        """Windows containing -inf values are also excluded."""
        n = 20
        tas = np.linspace(200, 250, n)
        tas[15] = -np.inf  # -inf in last window

        df = pl.DataFrame(
            {
                "alt": np.linspace(1000, 10000, n),
                "tas": tas,
                "cmd": np.zeros(n),
                "temp": np.full(n, 220.0),
                "d_alt": np.ones(n),
            }
        )
        path = tmp_path / "neginf.parquet"
        df.write_parquet(path)

        x, u, e, dx = self._cols()
        samples = _load_and_window([path], x, u, e, dx, seq_len=10, shift=10)
        # Second window (rows 10-19) contains -inf → should be filtered
        assert len(samples) == 1

    def test_segment_filter_fn(self, tmp_path: Path) -> None:
        """Custom segment_filter_fn can reject windows."""
        path = tmp_path / "seg.parquet"
        self._make_parquet(path)

        def reject_all(_df: pl.DataFrame, _start: int, _seq_len: int) -> bool:
            return False

        x, u, e, dx = self._cols()
        samples = _load_and_window(
            [path], x, u, e, dx, seq_len=10, shift=10, segment_filter_fn=reject_all
        )
        assert len(samples) == 0

    def test_preprocessing_fn(self, tmp_path: Path) -> None:
        """preprocessing_fn is applied before windowing."""
        n = 20
        df = pl.DataFrame(
            {
                "alt_raw": np.linspace(1000, 10000, n),
                "tas": np.linspace(200, 250, n),
                "cmd": np.zeros(n),
                "temp": np.full(n, 220.0),
                "d_alt": np.ones(n),
            }
        )
        path = tmp_path / "pp.parquet"
        df.write_parquet(path)

        def rename_alt(d: pl.DataFrame) -> pl.DataFrame:
            return d.rename({"alt_raw": "alt"})

        x, u, e, dx = self._cols()
        samples = _load_and_window(
            [path], x, u, e, dx, seq_len=10, shift=10, preprocessing_fn=rename_alt
        )
        assert len(samples) > 0
