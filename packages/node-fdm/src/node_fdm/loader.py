"""Helper for building train/validation datasets from a split DataFrame.

Reads flight parquet files, windows them into sequences, and returns
typed :class:`FlightDataset` instances.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import torch

from node_fdm.dataset import FlightDataset, FlightSample

__all__ = [
    "get_train_val_data",
]

log = structlog.get_logger("node_fdm.loader")


def _load_and_window(
    flight_paths: Sequence[str | Path],
    x_cols: list[str],
    u_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
    seq_len: int,
    shift: int,
    preprocessing_fn: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    segment_filter_fn: Callable[[pl.DataFrame, int, int], bool] | None = None,
) -> list[FlightSample]:
    """Load flights and slice into fixed-length windows.

    Args:
        flight_paths: Paths to parquet files.
        x_cols: State column names.
        u_cols: Control column names.
        e_cols: Environment column names.
        dx_cols: Derivative column names.
        seq_len: Window length.
        shift: Step between windows.
        preprocessing_fn: Optional preprocessing on the raw DataFrame.
        segment_filter_fn: Optional filter ``(df, start, seq_len) → bool``.

    Returns:
        List of windowed :class:`FlightSample` instances.
    """
    samples: list[FlightSample] = []
    all_cols = x_cols + u_cols + e_cols + dx_cols

    for path in flight_paths:
        df = pl.read_parquet(path)
        if preprocessing_fn is not None:
            df = preprocessing_fn(df)

        # Verify all columns exist
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            log.warning("missing_columns", path=str(path), missing=missing)
            continue

        n_rows = len(df)
        if n_rows < seq_len:
            continue

        # Extract arrays
        x_arr = df.select(x_cols).to_numpy().astype(np.float32)
        u_arr = df.select(u_cols).to_numpy().astype(np.float32)
        e_arr = df.select(e_cols).to_numpy().astype(np.float32)
        dx_arr = df.select(dx_cols).to_numpy().astype(np.float32)

        for start in range(0, n_rows - seq_len + 1, shift):
            end = start + seq_len

            # Check for NaN
            slices = [
                x_arr[start:end],
                u_arr[start:end],
                e_arr[start:end],
                dx_arr[start:end],
            ]
            if any(np.isnan(s).any() for s in slices):
                continue

            # Custom segment filter
            if segment_filter_fn is not None and not segment_filter_fn(df, start, seq_len):
                continue

            samples.append(
                FlightSample(
                    x=torch.from_numpy(slices[0].copy()),
                    u=torch.from_numpy(slices[1].copy()),
                    e=torch.from_numpy(slices[2].copy()),
                    dx=torch.from_numpy(slices[3].copy()),
                )
            )

    return samples


def get_train_val_data(
    data_df: pl.DataFrame,
    x_cols: list[str],
    u_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
    *,
    seq_len: int = 60,
    shift: int = 60,
    preprocessing_fn: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    segment_filter_fn: Callable[[pl.DataFrame, int, int], bool] | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> tuple[FlightDataset, FlightDataset]:
    """Create training and validation datasets from a labeled file list.

    Args:
        data_df: DataFrame with ``filepath`` and ``split`` columns.
        x_cols: State column names.
        u_cols: Control column names.
        e_cols: Environment column names.
        dx_cols: Derivative column names.
        seq_len: Window length for each sample.
        shift: Step between consecutive windows.
        preprocessing_fn: Optional flight preprocessing function.
        segment_filter_fn: Optional segment filter function.
        train_limit: Max number of training files to load.
        val_limit: Max number of validation files to load.

    Returns:
        Tuple of ``(train_dataset, val_dataset)``.
    """
    train_files = data_df.filter(pl.col("split") == "train").get_column("filepath").to_list()
    val_files = data_df.filter(pl.col("split") == "val").get_column("filepath").to_list()

    if train_limit is not None:
        train_files = train_files[:train_limit]
    if val_limit is not None:
        val_files = val_files[:val_limit]

    log.info("loading_data", train_files=len(train_files), val_files=len(val_files))

    train_samples = _load_and_window(
        train_files,
        x_cols,
        u_cols,
        e_cols,
        dx_cols,
        seq_len=seq_len,
        shift=shift,
        preprocessing_fn=preprocessing_fn,
        segment_filter_fn=segment_filter_fn,
    )
    val_samples = _load_and_window(
        val_files,
        x_cols,
        u_cols,
        e_cols,
        dx_cols,
        seq_len=seq_len,
        shift=shift,
        preprocessing_fn=preprocessing_fn,
        segment_filter_fn=segment_filter_fn,
    )

    log.info(
        "data_loaded",
        train_samples=len(train_samples),
        val_samples=len(val_samples),
    )

    return FlightDataset(train_samples), FlightDataset(val_samples)
