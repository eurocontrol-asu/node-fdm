"""Tests for node_fdm_data.preprocessing.qar — QAR preprocessing."""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_models.preprocessing.qar import (
    filter_noise,
    mode_stabilize,
    reduce_engine,
    smooth_strong,
)


class TestFilterNoise:
    """Butterworth low-pass filter tests."""

    def test_removes_high_freq(self) -> None:
        """High-frequency noise is attenuated."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 500)
        clean = np.sin(2 * np.pi * 2 * t)  # 2 Hz signal
        noisy = clean + 0.5 * rng.standard_normal(len(t))

        filtered = filter_noise(noisy, fs=500.0, cutoff=10.0, order=4)

        # Filtered should be closer to clean than noisy is
        noise_rmse = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        filtered_rmse = float(np.sqrt(np.mean((filtered - clean) ** 2)))
        assert filtered_rmse < noise_rmse

    def test_preserves_dc(self) -> None:
        """DC component (mean) is preserved."""
        signal = np.ones(100) * 5.0
        filtered = filter_noise(signal, fs=1.0, cutoff=0.1, order=2)
        np.testing.assert_allclose(filtered, 5.0, atol=0.01)

    def test_returns_ndarray(self) -> None:
        """Output is a numpy array with same length as input."""
        x = np.random.default_rng(0).standard_normal(50)
        result = filter_noise(x, fs=1.0, cutoff=0.1, order=2)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestSmoothStrong:
    """Moving-average smoothing tests."""

    def test_preserves_mean(self) -> None:
        """Smoothed signal has approximately the same mean as the original."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200) + 10.0
        smoothed = smooth_strong(x, window_size=20)
        np.testing.assert_allclose(np.mean(smoothed), np.mean(x), atol=0.5)

    def test_reduces_variance(self) -> None:
        """Smoothing reduces signal variance."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        smoothed = smooth_strong(x, window_size=20)
        assert float(np.var(smoothed)) < float(np.var(x))

    def test_same_length(self) -> None:
        """Output has same length as input."""
        x = np.ones(50)
        result = smooth_strong(x, window_size=10)
        assert len(result) == len(x)


class TestModeStabilize:
    """Transient-state suppression tests."""

    def test_suppresses_transients(self) -> None:
        """Short transient states are replaced by the previous stable state."""
        # Stable at 0, brief switch to 1 (3 samples), back to 0
        values = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        result = mode_stabilize(values, min_duration=5)
        # The brief 1s should be suppressed
        np.testing.assert_array_equal(result, np.zeros(13))

    def test_preserves_stable_change(self) -> None:
        """Sustained state changes are preserved."""
        values = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        result = mode_stabilize(values, min_duration=5)
        # After 5+ samples of 1, the mode should switch
        assert result[-1] == 1

    def test_returns_ndarray(self) -> None:
        """Output is an ndarray of same length."""
        values = np.array([1, 1, 2, 2, 3, 3])
        result = mode_stabilize(values, min_duration=2)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)


class TestReduceEngine:
    """Engine parameter reduction tests."""

    def test_symmetric_engines_average(self) -> None:
        """When engines are symmetric, result is the average."""
        df = pl.LazyFrame(
            {
                "n1_left": [80.0, 85.0],
                "n1_right": [80.0, 85.0],
            }
        )
        result = reduce_engine(df, col_left="n1_left", col_right="n1_right").collect()
        assert "n1" in result.columns
        assert result["n1"].to_list() == [80.0, 85.0]

    def test_asymmetric_engines_max(self) -> None:
        """When one engine is very low (< 5), result is the max."""
        df = pl.LazyFrame(
            {
                "n1_left": [80.0, 2.0],
                "n1_right": [3.0, 85.0],
            }
        )
        result = reduce_engine(df, col_left="n1_left", col_right="n1_right").collect()
        assert result["n1"].to_list() == [80.0, 85.0]

    def test_mixed_symmetric_asymmetric(self) -> None:
        """Mix of symmetric and asymmetric rows."""
        df = pl.LazyFrame(
            {
                "n1_left": [80.0, 3.0, 70.0],
                "n1_right": [82.0, 85.0, 72.0],
            }
        )
        result = reduce_engine(df, col_left="n1_left", col_right="n1_right").collect()
        expected = [81.0, 85.0, 71.0]  # avg, max, avg
        assert result["n1"].to_list() == expected
