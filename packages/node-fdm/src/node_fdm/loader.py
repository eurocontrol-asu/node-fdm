"""Helper for building train/validation datasets from a Delta Table DataFrame.

Reads flight data grouped by ``meta_flight_id``, windows them into
sequences, and returns typed :class:`FlightDataset` instances.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog
import torch

from node_fdm.dataset import FlightDataset, FlightSample

__all__ = [
    "get_train_val_data",
]

log = structlog.get_logger("node_fdm.loader")


def _fill_nan_sel(df: pl.DataFrame) -> pl.DataFrame:
    """Fill NaN and null to 0.0 on ``fdm_*_sel*`` columns."""
    sel_cols = [c for c in df.columns if c.startswith("fdm_") and "_sel" in c]
    if sel_cols:
        df = df.with_columns(
            [pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols],
        )
    return df


def _load_and_window(
    flights_df: pl.DataFrame,
    x_cols: list[str],
    u_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
    seq_len: int,
    shift: int,
    *,
    flight_limit: int | None = None,
    e1_cols: list[str] | None = None,
) -> list[FlightSample]:
    """Group flights and slice into fixed-length windows.

    Args:
        flights_df: DataFrame with all rows for one split, containing
            ``meta_flight_id`` and optionally ``fdm_flag_distance_ok``.
        x_cols: State column names.
        u_cols: Control column names.
        e_cols: Environment column names.
        dx_cols: Derivative column names.
        seq_len: Window length.
        shift: Step between windows.
        flight_limit: Max number of flights to process.
        e1_cols: Optional extra environment column names.

    Returns:
        List of windowed :class:`FlightSample` instances.
    """
    samples: list[FlightSample] = []
    all_cols = x_cols + u_cols + e_cols + dx_cols

    # Verify all columns exist
    missing = [c for c in all_cols if c not in flights_df.columns]
    if missing:
        log.warning("missing_columns", missing=missing)
        return samples

    # Resolve valid E1 columns (skip missing with warning)
    valid_e1_cols: list[str] = []
    if e1_cols:
        for col in e1_cols:
            if col in flights_df.columns:
                valid_e1_cols.append(col)
            else:
                log.warning("e1_column_missing", column=col)

    has_distance_flag = "fdm_flag_distance_ok" in flights_df.columns

    flight_ids = flights_df.get_column("meta_flight_id").unique().sort().to_list()
    if flight_limit is not None:
        flight_ids = flight_ids[:flight_limit]

    for fid in flight_ids:
        df = flights_df.filter(pl.col("meta_flight_id") == fid)
        n_rows = len(df)
        if n_rows < seq_len:
            continue

        # Extract arrays
        x_arr = df.select(x_cols).to_numpy().astype(np.float32)
        u_arr = df.select(u_cols).to_numpy().astype(np.float32)
        e_arr = df.select(e_cols).to_numpy().astype(np.float32)
        dx_arr = df.select(dx_cols).to_numpy().astype(np.float32)

        e1_arr: np.ndarray | None = None
        if valid_e1_cols:
            e1_arr = df.select(valid_e1_cols).to_numpy().astype(np.float32)

        # Distance flag array for segment filtering (AC6)
        dist_ok: np.ndarray | None = None
        if has_distance_flag:
            dist_ok = df.get_column("fdm_flag_distance_ok").to_numpy()

        for start in range(0, n_rows - seq_len + 1, shift):
            end = start + seq_len

            # Check for NaN / inf
            slices = [
                x_arr[start:end],
                u_arr[start:end],
                e_arr[start:end],
                dx_arr[start:end],
            ]
            if not all(np.isfinite(s).all() for s in slices):
                continue

            # Segment filter: all rows in window must have distance_ok (AC6)
            if dist_ok is not None and not dist_ok[start:end].all():
                continue

            e1_tensor: torch.Tensor | None = None
            if e1_arr is not None:
                e1_tensor = torch.from_numpy(e1_arr[start:end].copy())

            samples.append(
                FlightSample(
                    x=torch.from_numpy(slices[0].copy()),
                    u=torch.from_numpy(slices[1].copy()),
                    e=torch.from_numpy(slices[2].copy()),
                    dx=torch.from_numpy(slices[3].copy()),
                    e1=e1_tensor,
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
    train_limit: int | None = None,
    val_limit: int | None = None,
    e1_cols: list[str] | None = None,
) -> tuple[FlightDataset, FlightDataset]:
    """Create training and validation datasets from Delta Table data.

    The input DataFrame must contain ``meta_split`` and ``meta_flight_id``
    columns. Rows should already be filtered on ``fdm_flag_valid``.

    Fills NaN/null to 0.0 on ``fdm_*_sel*`` columns (U columns) at load
    time, so no separate preprocessing step is needed.

    Args:
        data_df: DataFrame loaded from the Delta Table, filtered on
            ``fdm_flag_valid`` and (optionally) ``meta_aircraft_type``.
        x_cols: State column names (SI units).
        u_cols: Control column names (SI units).
        e_cols: Environment column names (SI units).
        dx_cols: Derivative column names (SI units).
        seq_len: Window length for each sample.
        shift: Step between consecutive windows.
        train_limit: Max number of training flights to load.
        val_limit: Max number of validation flights to load.

    Returns:
        Tuple of ``(train_dataset, val_dataset)``.
    """
    # Fill NaN→0.0 on _sel columns (AC3)
    data_df = _fill_nan_sel(data_df)

    # Split by meta_split (AC2)
    train_df = data_df.filter(pl.col("meta_split") == "train")
    val_df = data_df.filter(pl.col("meta_split") == "val")

    n_train_flights = train_df.get_column("meta_flight_id").n_unique()
    n_val_flights = val_df.get_column("meta_flight_id").n_unique()
    log.info("loading_data", train_flights=n_train_flights, val_flights=n_val_flights)

    train_samples = _load_and_window(
        train_df,
        x_cols,
        u_cols,
        e_cols,
        dx_cols,
        seq_len=seq_len,
        shift=shift,
        flight_limit=train_limit,
        e1_cols=e1_cols,
    )
    val_samples = _load_and_window(
        val_df,
        x_cols,
        u_cols,
        e_cols,
        dx_cols,
        seq_len=seq_len,
        shift=shift,
        flight_limit=val_limit,
        e1_cols=e1_cols,
    )

    log.info(
        "data_loaded",
        train_samples=len(train_samples),
        val_samples=len(val_samples),
    )

    return FlightDataset(train_samples), FlightDataset(val_samples)
