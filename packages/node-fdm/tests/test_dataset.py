"""Tests for FlightSample, FlightDataset, and compute_stats."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from node_fdm.dataset import FlightDataset, FlightSample, compute_stats


def _make_sample(seq_len: int = 10) -> FlightSample:
    """Create a deterministic sample for testing."""
    return FlightSample(
        x=torch.randn(seq_len, 4),
        u=torch.randn(seq_len, 3),
        e=torch.randn(seq_len, 2),
        dx=torch.randn(seq_len, 4),
    )


class TestFlightSample:
    """Unit tests for the FlightSample frozen dataclass."""

    def test_frozen(self) -> None:
        """Assignment raises FrozenInstanceError."""
        sample = _make_sample()
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.x = torch.zeros(10, 4)  # type: ignore[misc]

    def test_fields(self) -> None:
        """All 4 tensor fields are present."""
        sample = _make_sample(seq_len=5)
        assert sample.x.shape == (5, 4)
        assert sample.u.shape == (5, 3)
        assert sample.e.shape == (5, 2)
        assert sample.dx.shape == (5, 4)


class TestFlightDataset:
    """Unit tests for FlightDataset."""

    def test_len(self) -> None:
        """Length matches input sample count."""
        samples = [_make_sample() for _ in range(7)]
        ds = FlightDataset(samples)
        assert len(ds) == 7

    def test_getitem_type(self) -> None:
        """__getitem__ returns a FlightSample instance."""
        ds = FlightDataset([_make_sample()])
        item = ds[0]
        assert isinstance(item, FlightSample)

    def test_empty_raises(self) -> None:
        """0 samples raises ValueError."""
        with pytest.raises(ValueError, match="at least one sample"):
            FlightDataset([])


class TestComputeStats:
    """Unit tests for compute_stats."""

    def test_keys(self) -> None:
        """Stats dict has expected column keys."""
        samples = [_make_sample() for _ in range(5)]
        x_cols = ["x1", "x2", "x3", "x4"]
        u_cols = ["u1", "u2", "u3"]
        e_cols = ["e1", "e2"]
        dx_cols = ["dx1", "dx2", "dx3", "dx4"]

        stats = compute_stats(samples, x_cols, u_cols, e_cols, dx_cols)
        expected_keys = set(x_cols + u_cols + e_cols + dx_cols)
        assert set(stats.keys()) == expected_keys

    def test_stat_fields(self) -> None:
        """Each column entry has mean, std, max."""
        samples = [_make_sample()]
        stats = compute_stats(
            samples,
            ["x1", "x2", "x3", "x4"],
            ["u1", "u2", "u3"],
            ["e1", "e2"],
            ["dx1", "dx2", "dx3", "dx4"],
        )
        for col_stats in stats.values():
            assert "mean" in col_stats
            assert "std" in col_stats
            assert "max" in col_stats
            # std should have the epsilon offset
            assert col_stats["std"] > 0

    def test_finite_values(self) -> None:
        """All stats are finite (no NaN/Inf)."""
        samples = [_make_sample() for _ in range(3)]
        stats = compute_stats(
            samples,
            ["x1", "x2", "x3", "x4"],
            ["u1", "u2", "u3"],
            ["e1", "e2"],
            ["dx1", "dx2", "dx3", "dx4"],
        )
        for col_stats in stats.values():
            for v in col_stats.values():
                assert not (v != v), f"NaN in stats: {col_stats}"  # NaN check
