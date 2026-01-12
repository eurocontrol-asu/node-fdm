#!/usr/bin/env python3
"""Helper for building train/validation datasets."""

from collections.abc import Callable
from typing import Any

import polars as pl

from node_fdm.data.dataset import SeqDataset


def get_train_val_data(
    data_df: pl.DataFrame,
    model_cols: Any,
    shift: int = 60,
    seq_len: int = 60,
    custom_fn: tuple[
        Callable[[pl.DataFrame], pl.DataFrame] | None,
        Callable[..., bool] | None,
    ] = (None, None),
    load_parallel: bool = True,
    train_val_num: tuple[int, int] = (5000, 500),
) -> tuple[SeqDataset, SeqDataset]:
    """Create training and validation datasets from a labeled file list.

    Args:
        data_df: DataFrame containing file paths with a `split` column.
        model_cols: Tuple containing model column groups.
        shift: Window shift used when generating sequences.
        seq_len: Sequence length for each sample.
        custom_fn: Tuple of optional processing and segment-filtering callables.
        load_parallel: Whether to load flights concurrently.
        train_val_num: Maximum number of train and validation files to load.

    Returns:
        Tuple of training and validation SeqDataset instances.
    """
    train_files = (
        data_df.filter(pl.col("split") == "train").get_column("filepath").to_list()
    )
    validation_files = (
        data_df.filter(pl.col("split") == "val").get_column("filepath").to_list()
    )

    train_dataset = SeqDataset(
        train_files[: train_val_num[0]],
        model_cols,
        seq_len=seq_len,
        shift=shift,
        custom_fn=custom_fn,
        load_parallel=load_parallel,
    )
    val_dataset = SeqDataset(
        validation_files[: train_val_num[1]],
        model_cols,
        seq_len=seq_len,
        shift=shift,
        custom_fn=custom_fn,
        load_parallel=load_parallel,
    )
    return train_dataset, val_dataset
