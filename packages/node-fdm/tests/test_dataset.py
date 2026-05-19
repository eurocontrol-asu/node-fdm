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


class TestComputeStatsExtra:
    """Tests for compute_stats with extra E1 columns (AXM-758)."""

    _x_cols: ClassVar[list[str]] = ["x1", "x2", "x3", "x4"]
    _u_cols: ClassVar[list[str]] = ["u1", "u2", "u3"]
    _e_cols: ClassVar[list[str]] = ["e1", "e2"]
    _dx_cols: ClassVar[list[str]] = ["dx1", "dx2", "dx3", "dx4"]

    @staticmethod
    def _make_sample_with_e1(
        seq_len: int = 10,
        n_e1: int = 3,
    ) -> FlightSample:
        """Create a sample with an extra e1 tensor."""
        return FlightSample(
            x=torch.randn(seq_len, 4),
            u=torch.randn(seq_len, 3),
            e=torch.randn(seq_len, 2),
            dx=torch.randn(seq_len, 4),
            e1=torch.randn(seq_len, n_e1),
        )

    def test_compute_stats_with_extra(self) -> None:
        """Extra e1 tensor (3 cols) produces stats with correct extra keys."""
        e1_cols = ["e1_wind", "e1_temp", "e1_press"]
        samples = [self._make_sample_with_e1(seq_len=20, n_e1=3) for _ in range(5)]

        stats = compute_stats(
            samples,
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
            e1_cols=e1_cols,
        )

        # All original + extra keys present
        expected_keys = set(self._x_cols + self._u_cols + self._e_cols + self._dx_cols + e1_cols)
        assert set(stats.keys()) == expected_keys

        # Extra cols have correct stat structure
        for col in e1_cols:
            assert "mean" in stats[col]
            assert "std" in stats[col]
            assert "max" in stats[col]

        # Verify values are computed from the e1 tensor, not zeros
        e1_all = torch.cat([s.e1 for s in samples if s.e1 is not None], dim=0)
        for i, col in enumerate(e1_cols):
            vals = e1_all[:, i]
            assert stats[col]["mean"] == pytest.approx(vals.mean().item(), abs=1e-4)
            assert stats[col]["std"] == pytest.approx(vals.std().item() + 1e-6, abs=1e-4)
            assert stats[col]["max"] == pytest.approx(vals.abs().max().item(), abs=1e-4)

    def test_compute_stats_without_extra(self) -> None:
        """Calling without e1_cols produces identical results to the original."""
        samples = [_make_sample(seq_len=10) for _ in range(5)]

        stats_original = compute_stats(
            samples,
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
        )
        stats_no_extra = compute_stats(
            samples,
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
            e1_cols=None,
        )

        assert stats_original == stats_no_extra

    def test_compute_stats_extra_no_overwrite(self) -> None:
        """Extra col with same name as existing col keeps original stats."""
        # Use "e1" which already exists in _e_cols
        e1_cols = ["e1"]
        samples = [
            FlightSample(
                x=torch.full((10, 4), 1.0),
                u=torch.full((10, 3), 1.0),
                e=torch.full((10, 2), 1.0),
                dx=torch.full((10, 4), 1.0),
                e1=torch.full((10, 1), 99.0),  # Different value
            )
            for _ in range(3)
        ]

        stats = compute_stats(
            samples,
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
            e1_cols=e1_cols,
        )

        # The e1 key should keep the base stats (1.0), not the e1 value (99.0)
        assert stats["e1"]["mean"] == pytest.approx(1.0, abs=1e-4)

    def test_compute_stats_empty_extra(self) -> None:
        """Empty extra_data tensor — stats dict unchanged."""
        samples = [_make_sample(seq_len=10) for _ in range(3)]

        stats_baseline = compute_stats(
            samples,
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
        )
        stats_empty = compute_stats(
            samples,
            self._x_cols,
            self._u_cols,
            self._e_cols,
            self._dx_cols,
            e1_cols=[],
        )

        assert stats_baseline == stats_empty


class TestE1StatsTasDiff:
    """Verify compute_stats covers fdm_tas_diff_ms via e1_cols (AXM-771)."""

    def test_e1_stats_include_tas_diff(self) -> None:
        """FlightSample list with e1 tensor including tas_diff → stats dict has fdm_tas_diff_ms."""
        seq_len = 20
        n_samples = 5
        # e1 tensor with 1 column representing fdm_tas_diff_ms
        samples = [
            FlightSample(
                x=torch.randn(seq_len, 4),
                u=torch.randn(seq_len, 3),
                e=torch.randn(seq_len, 2),
                dx=torch.randn(seq_len, 4),
                e1=torch.randn(seq_len, 1),
            )
            for _ in range(n_samples)
        ]

        stats = compute_stats(
            samples,
            ["x1", "x2", "x3", "x4"],
            ["u1", "u2", "u3"],
            ["e1", "e2"],
            ["dx1", "dx2", "dx3", "dx4"],
            e1_cols=["fdm_tas_diff_ms"],
        )

        assert "fdm_tas_diff_ms" in stats
        entry = stats["fdm_tas_diff_ms"]
        assert "mean" in entry
        assert "std" in entry
        assert "max" in entry
        # Values must be finite floats
        for key in ("mean", "std", "max"):
            v = entry[key]
            assert isinstance(v, float)
            assert v == v, f"{key} is NaN"  # NaN check
        # std includes epsilon so must be > 0
        assert entry["std"] > 0


class TestComputeStatsExtended:
    """Extended tests for compute_stats (mean/std, reverted from IQR in AXM-745)."""

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
        """Constant data — mean == constant value, std ≈ 1e-6."""
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
            assert stats[col]["mean"] == pytest.approx(val, abs=1e-4), (
                f"{col}: mean should be {val}, got {stats[col]['mean']}"
            )
            assert stats[col]["std"] == pytest.approx(1e-6, abs=1e-7), (
                f"{col}: std should be ~1e-6 for constant data, got {stats[col]['std']}"
            )

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
            assert stats[col]["std"] == pytest.approx(1e-6, abs=1e-7), (
                f"{col}: std should be epsilon (1e-6), got {stats[col]['std']}"
            )

    def test_compute_stats_zero_inflated(self) -> None:
        """90% zeros + 10% nonzero — std must NOT collapse to ~1e-6 (AXM-745).

        IQR-based std collapses for zero-inflated distributions because Q1=Q3=0.
        After reverting to mean/std, the standard deviation should reflect the
        actual spread of the data.
        """
        seq_len = 100
        n_samples = 10
        all_cols = self._x_cols + self._u_cols + self._e_cols + self._dx_cols
        samples: list[FlightSample] = []
        gen = torch.Generator().manual_seed(42)
        for _ in range(n_samples):
            # Build a row of mostly zeros with 10% nonzero values
            tensors: dict[str, torch.Tensor] = {}
            for name, cols in [
                ("x", self._x_cols),
                ("u", self._u_cols),
                ("e", self._e_cols),
                ("dx", self._dx_cols),
            ]:
                t = torch.zeros(seq_len, len(cols))
                # Set ~10% of rows to nonzero (value=5.0)
                mask = torch.rand(seq_len, generator=gen) < 0.1
                t[mask] = 5.0
                tensors[name] = t
            samples.append(
                FlightSample(
                    x=tensors["x"],
                    u=tensors["u"],
                    e=tensors["e"],
                    dx=tensors["dx"],
                )
            )
        stats = compute_stats(samples, self._x_cols, self._u_cols, self._e_cols, self._dx_cols)
        for col in all_cols:
            # With mean/std, std should be well above epsilon (~1.5 for this distribution)
            assert stats[col]["std"] > 0.1, (
                f"{col}: std={stats[col]['std']:.6f} collapsed to near-zero; "
                f"zero-inflated distribution needs real std, not IQR-based"
            )

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


def test_flight_sample_carries_flight_features() -> None:
    """AC1: FlightSample carries optional flight-level feature tensors."""
    flight_features = torch.zeros((4, 5))
    sample = FlightSample(
        x=torch.zeros((4, 2)),
        u=torch.zeros((4, 1)),
        e=torch.zeros((4, 1)),
        dx=torch.zeros((4, 2)),
        flight_features=flight_features,
    )
    default_sample = FlightSample(
        x=torch.zeros((4, 2)),
        u=torch.zeros((4, 1)),
        e=torch.zeros((4, 1)),
        dx=torch.zeros((4, 2)),
    )

    assert sample.flight_features is flight_features
    assert default_sample.flight_features is None


def test_compute_stats_flight_feature_cols() -> None:
    """AC2: compute_stats aggregates named flight feature columns positionally."""
    x_cols = ["era_tas_ms", "fdm_gamma_rad"]
    u_cols = ["cmd"]
    e_cols = ["rho"]
    dx_cols = ["fdm_d_tas_ms2", "fdm_d_gamma_rads"]
    flight_feature_cols = [
        "dist_total_flight",
        "dist_phase_flight",
        "mass_proxy_flight",
        "fuel_proxy_flight",
        "time_proxy_flight",
    ]
    samples = [
        FlightSample(
            x=torch.zeros((2, len(x_cols))),
            u=torch.zeros((2, len(u_cols))),
            e=torch.zeros((2, len(e_cols))),
            dx=torch.zeros((2, len(dx_cols))),
            flight_features=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]] * 2),
        )
        for _ in range(3)
    ]

    stats = compute_stats(
        samples,
        x_cols,
        u_cols,
        e_cols,
        dx_cols,
        flight_feature_cols=flight_feature_cols,
    )

    for index, col in enumerate(flight_feature_cols, start=1):
        assert stats[col]["mean"] == pytest.approx(float(index))
        assert stats[col]["std"] == pytest.approx(1e-6)
        assert stats[col]["max"] == pytest.approx(float(index))
        assert stats[col]["p999"] == pytest.approx(float(index))
        assert "iqr" not in stats[col]


def test_compute_stats_flight_feature_cols_neutral_fallback() -> None:
    """AC2: missing flight feature tensors produce neutral stats."""
    samples = [
        FlightSample(
            x=torch.zeros((2, 2)),
            u=torch.zeros((2, 1)),
            e=torch.zeros((2, 1)),
            dx=torch.zeros((2, 2)),
        )
    ]

    stats = compute_stats(
        samples,
        ["era_tas_ms", "fdm_gamma_rad"],
        ["cmd"],
        ["rho"],
        ["fdm_d_tas_ms2", "fdm_d_gamma_rads"],
        flight_feature_cols=["dist_total_flight", "mass_proxy_flight"],
    )

    neutral = {"mean": 0.0, "std": 1e-6, "max": 0.0, "p999": 0.0}
    assert stats["dist_total_flight"] == neutral
    assert stats["mass_proxy_flight"] == neutral


def test_derived_features_registers_hybrid_outputs() -> None:
    """Hybrid NN outputs are registered as derived features in DERIVED_FEATURES."""
    from node_fdm.dataset import DERIVED_FEATURES

    assert callable(DERIVED_FEATURES["fdm_t_minus_d_norm"])
    assert callable(DERIVED_FEATURES["fdm_lift_residual_norm"])


def test_t_minus_d_norm_computer_matches_a_spec() -> None:
    """``fdm_t_minus_d_norm`` collapses to ``a_spec = d_TAS + g*sin(gamma)`` for stats.

    Equivalent to the legacy ``a_spec`` target (m_ref / m factor uses m=m_ref).
    """
    import numpy as np

    from node_fdm.dataset import DERIVED_FEATURES
    from node_fdm_data.physics.constants import G

    x_cols = ["era_tas_ms", "fdm_gamma_rad"]
    e_cols = ["rho"]
    dx_cols = ["fdm_d_tas_ms2", "fdm_d_gamma_rads"]
    x_arr = np.array([[240.0, 0.05]], dtype=np.float64)
    e_arr = np.zeros((1, len(e_cols)), dtype=np.float64)
    dx_arr = np.array([[0.2, 0.001]], dtype=np.float64)

    result = DERIVED_FEATURES["fdm_t_minus_d_norm"](x_arr, e_arr, dx_arr, x_cols, e_cols, dx_cols)

    expected = 0.2 + G * np.sin(0.05)
    np.testing.assert_allclose(result[0], expected, rtol=0.0, atol=1e-5)


def test_lift_residual_norm_computer_matches_n_z_residual() -> None:
    """``fdm_lift_residual_norm`` collapses to ``n_z_residual`` for stats.

    The reconstruction ``L = lift_residual_norm · m_ref·g + m·g`` with
    ``m = m_ref`` makes the residual exactly ``n_z - 1``.
    """
    import numpy as np

    from node_fdm.dataset import DERIVED_FEATURES
    from node_fdm_data.physics.constants import G

    x_cols = ["era_tas_ms", "fdm_gamma_rad"]
    e_cols = ["rho"]
    dx_cols = ["fdm_d_tas_ms2", "fdm_d_gamma_rads"]
    x_arr = np.array([[240.0, 0.02]], dtype=np.float64)
    e_arr = np.zeros((1, len(e_cols)), dtype=np.float64)
    dx_arr = np.array([[0.2, 0.001]], dtype=np.float64)

    result = DERIVED_FEATURES["fdm_lift_residual_norm"](
        x_arr, e_arr, dx_arr, x_cols, e_cols, dx_cols
    )

    expected = (240.0 / G) * 0.001 + np.cos(0.02) - 1.0
    np.testing.assert_allclose(result[0], expected, rtol=0.0, atol=1e-5)


def test_compute_stats_derived_cols_includes_hybrid_outputs() -> None:
    """compute_stats derives p999 caps for hybrid adim output columns."""
    samples = [
        FlightSample(
            x=torch.tensor([[240.0, 0.02], [245.0, 0.03]]),
            u=torch.zeros((2, 0)),
            e=torch.ones((2, 1)),
            dx=torch.tensor([[0.2, 0.001], [0.25, 0.0015]]),
        )
    ]

    stats = compute_stats(
        samples,
        ["era_tas_ms", "fdm_gamma_rad"],
        [],
        ["rho"],
        ["fdm_d_tas_ms2", "fdm_d_gamma_rads"],
        derived_cols=["fdm_t_minus_d_norm", "fdm_lift_residual_norm"],
    )

    assert stats["fdm_t_minus_d_norm"]["p999"] > 0.0
    assert stats["fdm_lift_residual_norm"]["p999"] > 0.0
