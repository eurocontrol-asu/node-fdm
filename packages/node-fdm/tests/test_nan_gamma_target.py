"""Tests for AXM-816: loader and trainer must handle NaN gamma_target in e1 columns.

Bug: _load_and_window checks np.isfinite on x/u/e/dx but NOT on e1,
so NaN values in e1 columns (e.g. gamma_target) leak through to
compute_stats and ODETrainer, producing NaN statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from node_fdm.dataset import FlightDataset, FlightSample, compute_stats
from node_fdm.loader import _load_and_window

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df_with_e1(
    n: int = 30,
    flight_id: str = "f0",
    *,
    e1_nan_indices: list[int] | None = None,
    e1_inf_indices: list[int] | None = None,
) -> pl.DataFrame:
    """Create a flight DataFrame with an e1 column (gamma_target).

    Optionally inject NaN or inf at specific row indices.
    """
    gamma = np.ones(n, dtype=np.float64) * 0.05
    if e1_nan_indices:
        for idx in e1_nan_indices:
            gamma[idx] = np.nan
    if e1_inf_indices:
        for idx in e1_inf_indices:
            gamma[idx] = np.inf
    return pl.DataFrame(
        {
            "meta_flight_id": [flight_id] * n,
            "alt": np.linspace(1000, 10000, n).tolist(),
            "tas": np.linspace(200, 250, n).tolist(),
            "cmd": [0.0] * n,
            "temp": [220.0] * n,
            "d_alt": [1.0] * n,
            "gamma_target": gamma.tolist(),
        }
    )


_X = ["alt", "tas"]
_U = ["cmd"]
_E = ["temp"]
_DX = ["d_alt"]
_E1 = ["gamma_target"]


# ===========================================================================
# Unit tests — _load_and_window e1 NaN/inf filtering
# ===========================================================================


class TestLoadAndWindowE1NaN:
    """Windows with NaN/inf in e1 columns must be skipped."""

    def test_nan_in_e1_skips_window(self) -> None:
        """A window containing NaN in an e1 column is excluded."""
        df_nan = _make_df_with_e1(n=30, e1_nan_indices=[5])
        df_clean = _make_df_with_e1(n=30)

        nan_samples = _load_and_window(
            df_nan,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=5,
            e1_cols=_E1,
        )
        clean_samples = _load_and_window(
            df_clean,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=5,
            e1_cols=_E1,
        )
        assert len(nan_samples) < len(clean_samples)

    def test_inf_in_e1_skips_window(self) -> None:
        """A window containing inf in an e1 column is excluded."""
        df_inf = _make_df_with_e1(n=30, e1_inf_indices=[5])
        df_clean = _make_df_with_e1(n=30)

        inf_samples = _load_and_window(
            df_inf,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=5,
            e1_cols=_E1,
        )
        clean_samples = _load_and_window(
            df_clean,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=5,
            e1_cols=_E1,
        )
        assert len(inf_samples) < len(clean_samples)

    def test_all_e1_nan_produces_zero_samples(self) -> None:
        """When every row has NaN e1, no windows survive."""
        df = _make_df_with_e1(n=20, e1_nan_indices=list(range(20)))

        samples = _load_and_window(
            df,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=10,
            e1_cols=_E1,
        )
        assert len(samples) == 0

    def test_clean_e1_preserves_all_windows(self) -> None:
        """Clean e1 data should not reduce the number of windows."""
        df = _make_df_with_e1(n=30)

        samples_no_e1 = _load_and_window(
            df,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=10,
        )
        samples_with_e1 = _load_and_window(
            df,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=10,
            e1_cols=_E1,
        )
        assert len(samples_with_e1) == len(samples_no_e1)

    def test_e1_nan_only_affects_containing_window(self) -> None:
        """NaN in row 25 affects only the window that includes it, not earlier ones."""
        df = _make_df_with_e1(n=30, e1_nan_indices=[25])

        samples = _load_and_window(
            df,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=10,
            e1_cols=_E1,
        )
        # Windows: [0:10], [10:20], [20:30] — only last contains NaN at idx 25
        assert len(samples) == 2

    def test_neg_inf_in_e1_skips_window(self) -> None:
        """-inf in e1 is also non-finite and must be filtered."""
        n = 20
        gamma = np.ones(n) * 0.05
        gamma[15] = -np.inf
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": np.linspace(1000, 10000, n).tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
                "gamma_target": gamma.tolist(),
            }
        )

        samples = _load_and_window(
            df,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=10,
            e1_cols=_E1,
        )
        # Window [10:20] contains -inf at idx 15 → filtered
        assert len(samples) == 1


# ===========================================================================
# Unit tests — compute_stats with NaN in e1
# ===========================================================================


class TestComputeStatsE1NaN:
    """compute_stats must produce finite stats even when called defensively."""

    _x_cols: ClassVar[list[str]] = ["x1", "x2"]
    _u_cols: ClassVar[list[str]] = ["u1"]
    _e_cols: ClassVar[list[str]] = ["e1"]
    _dx_cols: ClassVar[list[str]] = ["dx1"]
    _e1_cols: ClassVar[list[str]] = ["gamma_target"]

    @staticmethod
    def _sample(*, e1_val: float = 0.05) -> FlightSample:
        return FlightSample(
            x=torch.randn(10, 2),
            u=torch.randn(10, 1),
            e=torch.randn(10, 1),
            dx=torch.randn(10, 1),
            e1=torch.full((10, 1), e1_val),
        )

    def test_clean_e1_stats_finite(self) -> None:
        """Clean e1 data → all stats are finite floats."""
        samples = [self._sample() for _ in range(5)]
        stats = compute_stats(
            samples,
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
            e1_cols=self._e1_cols,
        )
        assert "gamma_target" in stats
        for key in ("mean", "std", "max"):
            v = stats["gamma_target"][key]
            assert isinstance(v, float)
            assert v == v, f"gamma_target.{key} is NaN"

    def test_nan_e1_stats_still_finite(self) -> None:
        """If NaN somehow reaches compute_stats in e1, stats must not be NaN.

        After the fix in _load_and_window this scenario should not happen,
        but compute_stats should be defensive.
        """
        e1_tensor = torch.full((10, 1), 0.05)
        e1_tensor[3, 0] = float("nan")
        sample = FlightSample(
            x=torch.randn(10, 2),
            u=torch.randn(10, 1),
            e=torch.randn(10, 1),
            dx=torch.randn(10, 1),
            e1=e1_tensor,
        )
        stats = compute_stats(
            [sample],
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
            e1_cols=self._e1_cols,
        )
        assert "gamma_target" in stats
        for key in ("mean", "std", "max"):
            v = stats["gamma_target"][key]
            assert v == v, f"gamma_target.{key} is NaN — compute_stats must handle NaN in e1"


# ===========================================================================
# Functional test — full pipeline e1 NaN filtering
# ===========================================================================


class TestE1NaNPipeline:
    """End-to-end: loader → dataset → compute_stats with NaN e1."""

    def test_loader_to_stats_with_nan_e1(self) -> None:
        """Full path: DataFrame with NaN gamma_target → stats are finite."""
        # Mix of clean and NaN flights
        df_clean = _make_df_with_e1(n=50, flight_id="f0")
        df_nan = _make_df_with_e1(n=50, flight_id="f1", e1_nan_indices=[10, 20, 30])
        df = pl.concat([df_clean, df_nan])

        samples = _load_and_window(
            df,
            _X,
            _U,
            _E,
            _DX,
            seq_len=10,
            shift=10,
            e1_cols=_E1,
        )
        assert len(samples) > 0

        # All surviving samples must have finite e1
        for s in samples:
            assert s.e1 is not None
            assert s.e1.isfinite().all(), "NaN/inf leaked through _load_and_window into e1"

        # compute_stats should produce finite results
        stats = compute_stats(
            samples,
            _X,
            _U,
            _E,
            _DX,
            e1_cols=_E1,
        )
        assert "gamma_target" in stats
        for key in ("mean", "std", "max"):
            v = stats["gamma_target"][key]
            assert v == v, f"gamma_target.{key} is NaN after full pipeline"


# ===========================================================================
# ODETrainer.__init__ — e1_cols forwarded to compute_stats
# ===========================================================================


class TestTrainerE1ColsForwarding:
    """ODETrainer.__init__ must pass e1_cols to compute_stats."""

    def test_trainer_passes_e1_cols(self, tmp_path: Path) -> None:
        """compute_stats is called with e1_cols from the architecture spec."""
        from node_fdm.trainer import ODETrainer

        # We patch compute_stats to capture the kwargs it receives
        calls: list[dict[str, object]] = []
        original_compute_stats = compute_stats

        def spy_compute_stats(*args, **kwargs):
            calls.append(kwargs)
            return original_compute_stats(*args, **kwargs)

        # Build minimal valid datasets
        samples = [
            FlightSample(
                x=torch.randn(10, 2),
                u=torch.randn(10, 1),
                e=torch.randn(10, 1),
                dx=torch.randn(10, 1),
                e1=torch.randn(10, 1),
            )
            for _ in range(3)
        ]
        ds = FlightDataset(samples)

        mock_model = torch.nn.Linear(1, 1)  # Provides real parameters

        with (
            patch("node_fdm.trainer.compute_stats", side_effect=spy_compute_stats),
            patch("node_fdm.trainer.get") as mock_get,
            patch("node_fdm.trainer.FlightDynamicsModel", return_value=mock_model),
            patch("node_fdm.trainer.get_loss"),
            patch.object(
                ODETrainer,
                "_build_norm_vectors",
                return_value=(torch.zeros(1), torch.ones(1)),
            ),
            patch.object(ODETrainer, "_build_alpha_weights", return_value=torch.ones(1)),
            patch.object(ODETrainer, "save_meta"),
        ):
            # Configure mock spec with e1_cols
            mock_spec = mock_get.return_value
            mock_spec.x_cols = ["alt", "tas"]
            mock_spec.u_cols = ["cmd"]
            mock_spec.e0_cols = ["temp"]
            mock_spec.e1_cols = ["gamma_target"]
            mock_spec.dx_cols = [("d_alt", "d_alt")]

            from node_fdm.trainer import TrainingConfig

            config = TrainingConfig(
                architecture_name="test",
                model_name="test_model",
            )

            ODETrainer(
                config=config,
                train_dataset=ds,
                val_dataset=ds,
                model_dir=tmp_path / "test_trainer",
            )

        assert len(calls) >= 1, "compute_stats was not called"
        assert "e1_cols" in calls[0], "ODETrainer.__init__ must pass e1_cols to compute_stats"
        assert calls[0]["e1_cols"] == ["gamma_target"]
