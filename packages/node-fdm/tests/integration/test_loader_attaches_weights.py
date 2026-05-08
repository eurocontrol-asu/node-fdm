from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

pytestmark = pytest.mark.integration

from node_fdm.loader import _load_and_window  # noqa: E402


def _build_df(n_rows: int = 120, weight: float = 2.5) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "meta_flight_id": ["F1"] * n_rows,
            "x0": rng.standard_normal(n_rows).astype(np.float32),
            "u0": rng.standard_normal(n_rows).astype(np.float32),
            "e0": rng.standard_normal(n_rows).astype(np.float32),
            "dx0": rng.standard_normal(n_rows).astype(np.float32),
            "fdm_train_weight": np.full(n_rows, weight, dtype=np.float32),
        }
    )


def test_loader_carries_weights_from_dataframe_to_flight_sample() -> None:
    df = _build_df()
    samples = _load_and_window(
        df,
        x_cols=["x0"],
        u_cols=["u0"],
        e_cols=["e0"],
        dx_cols=["dx0"],
        seq_len=60,
        shift=60,
    )
    assert samples, "loader produced no windows"
    for s in samples:
        assert s.w is not None
        assert s.w.shape == (60,)
        assert torch.allclose(s.w, torch.full((60,), 2.5))
