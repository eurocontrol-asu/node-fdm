"""Tests for FlightSample, FlightDataset, and compute_stats."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

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


class TestComputeStatsRobust:
    """Tests for robust median/IQR statistics in compute_stats (AXM-741)."""

    _x_cols: ClassVar[list[str]] = ["x1", "x2", "x3", "x4"]
    _u_cols: ClassVar[list[str]] = ["u1", "u2", "u3"]
    _e_cols: ClassVar[list[str]] = ["e1", "e2"]
    _dx_cols: ClassVar[list[str]] = ["dx1", "dx2", "dx3", "dx4"]

    def test_compute_stats_keys(self) -> None:
        """10 normal samples — dict contains 'mean', 'std', 'max' for each column."""
        samples = [_make_sample(seq_len=10) for _ in range(10)]
        stats = compute_stats(samples, self._x_cols, self._u_cols, self._e_cols, self._dx_cols)
        all_cols = self._x_cols + self._u_cols + self._e_cols + self._dx_cols
        assert set(stats.keys()) == set(all_cols)
        for col in all_cols:
            assert "mean" in stats[col], f"missing 'mean' for {col}"
            assert "std" in stats[col], f"missing 'std' for {col}"
            assert "max" in stats[col], f"missing 'max' for {col}"

    def test_compute_stats_values_constant(self) -> None:
        """Constant data — median == constant value, std ≈ 1e-6."""
        val = 42.0
        samples = [
            FlightSample(
                x=torch.full((10, len(self._x_cols)), val),
                u=torch.full((10, len(self._u_cols)), val),
                e=torch.full((10, len(self._e_cols)), val),
                dx=torch.full((10, len(self._dx_cols)), val),
            )
            for _ in range(5)
        ]
        stats = compute_stats(samples, self._x_cols, self._u_cols, self._e_cols, self._dx_cols)
        for col in self._x_cols + self._u_cols + self._e_cols + self._dx_cols:
            assert stats[col]["mean"] == pytest.approx(
                val, abs=1e-4
            ), f"{col}: median should be {val}, got {stats[col]['mean']}"
            assert stats[col]["std"] == pytest.approx(
                1e-6, abs=1e-7
            ), f"{col}: std should be ~1e-6 for constant data, got {stats[col]['std']}"

    def test_outlier_extreme_robust_mean(self) -> None:
        """99 samples at ~100, 1 sample at 50000 — 'mean' ≈ 100 (robust, not ~600)."""
        normal_val = 100.0
        outlier_val = 50000.0
        seq_len = 10
        # 99 normal samples
        normal_samples = [
            FlightSample(
                x=torch.full((seq_len, len(self._x_cols)), normal_val),
                u=torch.full((seq_len, len(self._u_cols)), normal_val),
                e=torch.full((seq_len, len(self._e_cols)), normal_val),
                dx=torch.full((seq_len, len(self._dx_cols)), normal_val),
            )
            for _ in range(99)
        ]
        # 1 outlier sample
        outlier_sample = FlightSample(
            x=torch.full((seq_len, len(self._x_cols)), outlier_val),
            u=torch.full((seq_len, len(self._u_cols)), outlier_val),
            e=torch.full((seq_len, len(self._e_cols)), outlier_val),
            dx=torch.full((seq_len, len(self._dx_cols)), outlier_val),
        )
        samples = [*normal_samples, outlier_sample]
        stats = compute_stats(samples, self._x_cols, self._u_cols, self._e_cols, self._dx_cols)
        for col in self._x_cols + self._u_cols + self._e_cols + self._dx_cols:
            # Robust mean (median) should be ~100, not pulled up to ~600 by outlier
            assert stats[col]["mean"] == pytest.approx(
                normal_val, rel=0.05
            ), f"{col}: robust mean should be ~{normal_val}, got {stats[col]['mean']}"
            # Std should reflect the healthy distribution, not be inflated by outlier
            assert (
                stats[col]["std"] < 500
            ), f"{col}: std should reflect sane distribution, got {stats[col]['std']}"

    def test_constant_column_std_epsilon(self) -> None:
        """All identical values — std = 1e-6 (not 0)."""
        val = 7.0
        samples = [
            FlightSample(
                x=torch.full((20, len(self._x_cols)), val),
                u=torch.full((20, len(self._u_cols)), val),
                e=torch.full((20, len(self._e_cols)), val),
                dx=torch.full((20, len(self._dx_cols)), val),
            )
            for _ in range(10)
        ]
        stats = compute_stats(samples, self._x_cols, self._u_cols, self._e_cols, self._dx_cols)
        for col in self._x_cols + self._u_cols + self._e_cols + self._dx_cols:
            assert stats[col]["std"] > 0, f"{col}: std must not be zero"
            assert stats[col]["std"] == pytest.approx(
                1e-6, abs=1e-7
            ), f"{col}: std should be epsilon (1e-6), got {stats[col]['std']}"

    def test_single_sample(self) -> None:
        """1 sample with seq_len=60 — no crash, stats calculated."""
        samples = [
            FlightSample(
                x=torch.randn(60, len(self._x_cols)),
                u=torch.randn(60, len(self._u_cols)),
                e=torch.randn(60, len(self._e_cols)),
                dx=torch.randn(60, len(self._dx_cols)),
            )
        ]
        stats = compute_stats(samples, self._x_cols, self._u_cols, self._e_cols, self._dx_cols)
        all_cols = self._x_cols + self._u_cols + self._e_cols + self._dx_cols
        assert set(stats.keys()) == set(all_cols)
        for col in all_cols:
            for key in ("mean", "std", "max"):
                v = stats[col][key]
                assert isinstance(v, float), f"{col}.{key} should be float"
                assert not (v != v), f"{col}.{key} is NaN"
