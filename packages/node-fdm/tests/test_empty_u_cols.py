"""Tests for AXM-815: compute_stats and loader crash when U_ODE_COLS is empty.

Validates that:
- compute_stats handles u_cols=[] with zero-width u tensors (no crash, no U entries).
- compute_stats still works correctly with non-empty u_cols (backward compat).
- _load_and_window produces FlightSample.u with shape (seq, 0) when u_cols=[].
- FlightDynamicsModel forward pass succeeds with NODE_ADSB_V1 spec (empty U_ODE_COLS).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from node_fdm.architectures.registry import get
from node_fdm.dataset import FlightSample, compute_stats
from node_fdm.loader import _load_and_window
from node_fdm.models.fdm import FlightDynamicsModel

# ---------------------------------------------------------------------------
# compute_stats — empty u_cols
# ---------------------------------------------------------------------------


class TestComputeStatsEmptyU:
    """compute_stats must not crash when u_cols is empty."""

    def test_compute_stats_empty_u_cols(self) -> None:
        """Samples with u shape (seq, 0), u_cols=[] → stats has no U entries, no crash."""
        seq_len = 10
        samples = [
            FlightSample(
                x=torch.randn(seq_len, 3),
                u=torch.empty(seq_len, 0),
                e=torch.randn(seq_len, 2),
                dx=torch.randn(seq_len, 3),
            )
            for _ in range(5)
        ]
        x_cols = ["x1", "x2", "x3"]
        u_cols: list[str] = []
        e_cols = ["e1", "e2"]
        dx_cols = ["dx1", "dx2", "dx3"]

        stats = compute_stats(samples, x_cols, u_cols, e_cols, dx_cols)

        # No U keys in output
        expected_keys = set(x_cols + e_cols + dx_cols)
        assert set(stats.keys()) == expected_keys

        # All stats are finite
        for col_stats in stats.values():
            for v in col_stats.values():
                assert isinstance(v, float)
                assert v == v, f"NaN in stats: {col_stats}"

    def test_compute_stats_nonempty_u_cols(self) -> None:
        """u shape (seq, 2), u_cols=[a, b] → stats correct (backward compat)."""
        seq_len = 10
        val = 5.0
        samples = [
            FlightSample(
                x=torch.randn(seq_len, 3),
                u=torch.full((seq_len, 2), val),
                e=torch.randn(seq_len, 2),
                dx=torch.randn(seq_len, 3),
            )
            for _ in range(5)
        ]
        x_cols = ["x1", "x2", "x3"]
        u_cols = ["u_a", "u_b"]
        e_cols = ["e1", "e2"]
        dx_cols = ["dx1", "dx2", "dx3"]

        stats = compute_stats(samples, x_cols, u_cols, e_cols, dx_cols)

        # U keys present
        assert "u_a" in stats
        assert "u_b" in stats

        # U stats reflect constant value
        for col in u_cols:
            assert stats[col]["mean"] == float(torch.tensor(val).mean()), f"{col} mean mismatch"
            assert stats[col]["std"] > 0  # epsilon offset


# ---------------------------------------------------------------------------
# _load_and_window — empty u_cols
# ---------------------------------------------------------------------------


class TestLoadAndWindowEmptyU:
    """_load_and_window must produce zero-width u tensors when u_cols is empty."""

    def test_load_and_window_empty_u(self) -> None:
        """DataFrame with no U columns, u_cols=[] → FlightSample.u has shape (seq, 0)."""
        n = 50
        seq_len = 10
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": np.linspace(1000, 10000, n).tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
            }
        )

        samples = _load_and_window(
            df,
            x_cols=["alt", "tas"],
            u_cols=[],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=seq_len,
            shift=seq_len,
        )

        assert len(samples) > 0
        for s in samples:
            assert s.u.shape == (seq_len, 0), f"Expected (seq, 0), got {s.u.shape}"
            assert s.x.shape == (seq_len, 2)
            assert s.e.shape == (seq_len, 1)
            assert s.dx.shape == (seq_len, 1)


# ---------------------------------------------------------------------------
# FlightDynamicsModel.forward — NODE_ADSB_V1 (empty U_ODE_COLS)
# ---------------------------------------------------------------------------


def _make_stats(cols: list[str]) -> dict[str, dict[str, float]]:
    """Build a dummy stats_dict covering all columns."""
    return {col: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 0.8} for col in cols}


class TestForwardPassEmptyUOde:
    """Forward pass succeeds with NODE_ADSB_V1 spec (U_ODE_COLS is empty)."""

    def test_forward_pass_empty_u_ode(self) -> None:
        """NODE_ADSB_V1 spec, dummy stats → forward pass succeeds."""
        spec = get("node_adsb_v1")
        dx_col_names = [c for _, c in spec.dx_cols]
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols + dx_col_names
        all_cols += spec.derived_output_cols
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        batch = 4
        x = torch.randn(batch, len(spec.x_cols))
        u = torch.randn(batch, len(spec.u_cols))
        e = torch.randn(batch, len(spec.e0_cols))

        out = model(x, u, e)

        assert out.shape == (batch, len(spec.dx_cols))
        assert torch.isfinite(out).all(), "Forward output contains NaN/Inf"
