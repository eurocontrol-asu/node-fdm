#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset utilities for loading and segmenting flight data sequences."""

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from node_fdm.data.flight_processor import FlightProcessor


class SeqDataset(Dataset):
    """Sequence dataset that loads flight segments for model training."""

    def __init__(
        self,
        flights_path_list: Sequence[str],
        model_cols: Tuple[Any, Any, Any, Any, Any],
        seq_len: int = 60,
        shift: int = 60,
        n_jobs: int = 35,
        load_parallel: bool = True,
        custom_fn: Tuple[
            Optional[Callable[[pd.DataFrame], pd.DataFrame]],
            Optional[Callable[..., bool]],
        ] = (None, None),
    ) -> None:
        """Initialize the dataset with flight paths and model column definitions.

        Args:
            flights_path_list: Iterable of flight parquet file paths.
            model_cols: Tuple containing model column groups (state, control, env, etc.).
            seq_len: Sequence length to extract from each flight.
            shift: Step size when sliding the sequence window.
            n_jobs: Number of parallel workers to use when loading flights.
            load_parallel: Whether to load flights concurrently.
            custom_fn: Tuple of optional processing and segment-filtering callables.
        """
        self.flights_path_list = flights_path_list
        self.shift = shift
        self.seq_len = seq_len
        self.x_cols, self.u_cols, self.e0_cols, self.e_cols, self.dx_cols = model_cols
        self.deriv_cols = [col.derivative for col in self.x_cols]
        self.model_cols = model_cols
        self.load_parallel = load_parallel
        self.n_jobs = n_jobs
        custom_processing_fn, custom_segment_filtering_fn = custom_fn
        self.processor = FlightProcessor(
            model_cols, custom_processing_fn=custom_processing_fn
        )
        self.custom_segment_filtering_fn = custom_segment_filtering_fn
        self.init_flight_date()

    def init_flight_date(self) -> None:
        """Load all flights, build sequence cache, and compute aggregate statistics.

        Populates internal sequence list and per-column statistics used for normalization.
        """
        if self.load_parallel:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_one_flight)(
                    flight,
                )
                for flight in tqdm(self.flights_path_list, desc="Loading flights")
            )
        else:
            results = [
                self.process_one_flight(flight)
                for flight in tqdm(self.flights_path_list, desc="Loading flights")
            ]

        self.sequences = []
        for seqs in results:
            self.sequences.extend(seqs)

        all_data = np.concatenate(
            [
                np.concatenate([seq[0], seq[1], seq[2], seq[3]], axis=1)
                for seq in self.sequences
            ],
            axis=0,
        )

        all_cols = (
            self.x_cols + self.u_cols + self.e0_cols + self.e_cols + self.deriv_cols
        )

        self.stats_dict = dict()

        for i, col in enumerate(all_cols):
            vals = all_data[:, i].astype(float)
            self.stats_dict[col] = {
                "mean": vals.mean(),
                "std": vals.std() + 1e-6,
                "max": np.percentile(np.abs(vals), 99.5),
            }

    def process_one_flight(
        self, flight_path: str
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Process a single flight file into clean, nan-free sequences.

        Args:
            flight_path: Path to a flight parquet file.

        Returns:
            List of tuples containing state, control, environment, and derivative arrays.
        """
        f = self.read_flight(flight_path)
        seqs = []
        N = len(f)
        if N > self.seq_len:
            x_seq = f[self.x_cols].values.astype(np.float32)
            u_seq = f[self.u_cols].values.astype(np.float32)
            e_seq = f[self.e0_cols + self.e_cols].values.astype(np.float32)
            dx_seq = f[self.deriv_cols].values.astype(np.float32)

            for start in range(0, N - self.seq_len + 1, self.shift):
                custom_segment_filtering_bool = True
                if self.custom_segment_filtering_fn is not None:
                    custom_segment_filtering_bool = self.custom_segment_filtering_fn(
                        f, start, self.seq_len
                    )
                nans = sum(
                    [
                        np.isnan(seq[start : start + self.seq_len]).sum()
                        for seq in [x_seq, u_seq, e_seq, dx_seq]
                    ]
                )
                if (custom_segment_filtering_bool) & (nans == 0):
                    seqs.append(
                        (
                            x_seq[start : start + self.seq_len],
                            u_seq[start : start + self.seq_len],
                            e_seq[start : start + self.seq_len],
                            dx_seq[start : start + self.seq_len],
                        )
                    )
        return seqs

    def __len__(self) -> int:
        """Return number of available sequences.

        Returns:
            Count of cached flight sequences.
        """
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tensors for a specific sequence index.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            Tuple of tensors for state, control, environment, and derivative slices.
        """
        x_seq, u_seq, e_seq, dxdt_seq = self.sequences[idx]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(u_seq, dtype=torch.float32),
            torch.tensor(e_seq, dtype=torch.float32),
            torch.tensor(dxdt_seq, dtype=torch.float32),
        )

    def read_flight(self, flight_path: str) -> pd.DataFrame:
        """Read a flight parquet file and apply base processing.

        Args:
            flight_path: Path to a parquet file containing flight data.

        Returns:
            Processed DataFrame with standardized columns.
        """
        f = pd.read_parquet(flight_path)
        return self.processor.process_flight(f)

    def get_full_flight(
        self, flight_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Return full arrays for a specific flight index.

        Args:
            flight_idx: Index of the flight in the provided flight list.

        Returns:
            Tuple of state, control, environment, derivative arrays, and the full DataFrame.
        """
        flight_path = self.flights_path_list[flight_idx]
        f = self.read_flight(flight_path)
        x_seq = f[self.x_cols].values.astype(np.float32)
        u_seq = f[self.u_cols].values.astype(np.float32)
        e0_seq = f[self.e0_cols + self.e_cols].values.astype(np.float32)
        dx_seq = f[self.deriv_cols].values.astype(np.float32)
        return x_seq, u_seq, e0_seq, dx_seq, f
