
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from node_fdm.architectures.opensky_2025.columns import col_dist
from node_fdm.data.flight_processor import FlightProcessor

from joblib import Parallel, delayed

from config import DEFAULT_CPU_COUNT


class SeqDataset(Dataset):
    def __init__(
        self,
        flights_path_list,
        model_cols,
        seq_len=60,
        shift=60,
        n_jobs=DEFAULT_CPU_COUNT,
        load_parallel=True,
        custom_fn = (None,None)
    ):
        self.flights_path_list = flights_path_list
        self.shift = shift
        self.seq_len = seq_len
        self.x_cols, self.u_cols, self.e0_cols, self.e_cols, self.dx_cols = model_cols
        self.model_cols = model_cols
        self.load_parallel = load_parallel
        self.n_jobs = n_jobs
        custom_processing_fn, custom_segment_filtering_fn = custom_fn
        self.processor = FlightProcessor(model_cols, custom_processing_fn=custom_processing_fn)
        self.custom_segment_filtering_fn = custom_segment_filtering_fn
        self.init_flight_date()

    def init_flight_date(self):

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

        all_cols = self.x_cols + self.u_cols + self.e0_cols + self.e_cols + self.dx_cols

        self.stats_dict = dict()

        for i, col in enumerate(all_cols):
            vals = all_data[:, i].astype(float)
            self.stats_dict[col] = {
                "mean": vals.mean(),
                "std": vals.std() + 1e-6,
                "max": np.percentile(np.abs(vals), 99.5),
            }

    def process_one_flight(self, flight_path):
        f = self.read_flight(flight_path)
        seqs = []
        N = len(f)
        if N > self.seq_len:
            x_seq = f[self.x_cols].values.astype(np.float32)
            u_seq = f[self.u_cols].values.astype(np.float32)
            e_seq = f[self.e0_cols + self.e_cols].values.astype(np.float32)
            dx_seq = f[self.dx_cols].values.astype(np.float32)

            for start in range(0, N - self.seq_len + 1, self.shift):
                custom_segment_filtering_bool= True
                if self.custom_segment_filtering_fn is not None:
                    custom_segment_filtering_bool = self.custom_segment_filtering_fn(f, start, self.seq_len)
                nans = sum([np.isnan(seq[start:start+self.seq_len]).sum() for seq in [x_seq, u_seq, e_seq, dx_seq]])
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, u_seq, e_seq, dxdt_seq = self.sequences[idx]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(u_seq, dtype=torch.float32),
            torch.tensor(e_seq, dtype=torch.float32),
            torch.tensor(dxdt_seq, dtype=torch.float32),
        )

    def read_flight(self, flight_path):
        f = pd.read_parquet(flight_path)
        return self.processor.process_flight(f)

    def get_full_flight(self, flight_idx):
        """
        Returns full arrays (n_points, n_state), (n_points, n_ctrl), (n_points, n_env)
        for flight_idx in self.flights.
        """
        flight_path = self.flights_path_list[flight_idx]
        f = self.read_flight(flight_path)
        x_seq = f[self.x_cols].values.astype(np.float32)
        u_seq = f[self.u_cols].values.astype(np.float32)
        e0_seq = f[self.e0_cols + self.e_cols].values.astype(np.float32)
        dx_seq = f[self.dx_cols].values.astype(np.float32)
        return x_seq, u_seq, e0_seq, dx_seq, f
