"""Tests for compute_stats derived feature computation (kinematic e1 columns).

These features are produced at runtime by TrajectoryLayer and are NOT
present in the parquet dataset. compute_stats must compute them analytically
from s.x / s.e so that InputNormalizer gets proper mean/std.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from node_fdm.dataset import DERIVED_FEATURES, FlightSample, compute_stats
from node_fdm_data.physics.constants import G, R
from node_fdm_data.physics.isa import isa_pressure

# ── column names matching adsb architecture ──────────────────────────────────
_X_COLS = ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]
_U_COLS: list[str] = []
_E_COLS = ["fdm_long_wind_ms", "era_temp_K"]
_DX_COLS = ["fdm_d_alt_ms", "fdm_d_gamma_rads", "fdm_d_tas_ms2"]

_DERIVED_E1_COLS = [
    "fdm_g_sin_gamma_ms2",
    "fdm_cos_gamma",
    "fdm_g_over_v",
    "fdm_q_pa",
]


def _make_sample(
    *,
    alt_m: float = 5000.0,
    gamma_rad: float = 0.05,
    tas_ms: float = 200.0,
    temp_k: float = 255.65,
    seq_len: int = 20,
    include_e1: bool = False,
    n_e1: int = 4,
) -> FlightSample:
    """Build a synthetic sample with known physical values."""
    x = torch.tensor(
        [[alt_m, gamma_rad, tas_ms]] * seq_len,
        dtype=torch.float32,
    )
    u = torch.zeros(seq_len, 0)  # empty u
    e = torch.tensor(
        [[0.0, temp_k]] * seq_len,  # [wind, temp]
        dtype=torch.float32,
    )
    dx = torch.zeros(seq_len, 3)
    e1 = torch.zeros(seq_len, n_e1) if include_e1 else None
    return FlightSample(x=x, u=u, e=e, dx=dx, e1=e1)


class TestDerivedFeaturesRegistry:
    """Unit: DERIVED_FEATURES registry has the four expected keys."""

    def test_registry_keys(self) -> None:
        """All four derived feature keys must be present."""
        expected = {"fdm_g_sin_gamma_ms2", "fdm_cos_gamma", "fdm_g_over_v", "fdm_q_pa"}
        assert set(DERIVED_FEATURES.keys()) >= expected

    def test_registry_callables(self) -> None:
        """Every entry in DERIVED_FEATURES must be callable."""
        for name, fn in DERIVED_FEATURES.items():
            assert callable(fn), f"{name} is not callable"


class TestComputeStatsDerivedColumns:
    """Unit: compute_stats returns stats for all 4 derived features when they
    are listed in e1_cols but NOT present in s.e1.
    """

    def test_all_four_derived_columns_present(self) -> None:
        """Stats dict contains all 4 derived feature keys."""
        samples = [_make_sample() for _ in range(5)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=_DERIVED_E1_COLS,
        )
        for col in _DERIVED_E1_COLS:
            assert col in stats, f"Missing stats for {col}"

    def test_derived_stats_structure(self) -> None:
        """Each derived feature entry has mean, std, max."""
        samples = [_make_sample() for _ in range(3)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=_DERIVED_E1_COLS,
        )
        for col in _DERIVED_E1_COLS:
            entry = stats[col]
            assert "mean" in entry, f"{col}: missing 'mean'"
            assert "std" in entry, f"{col}: missing 'std'"
            assert "max" in entry, f"{col}: missing 'max'"

    def test_derived_stats_std_positive(self) -> None:
        """std > 0 (epsilon floor applied)."""
        samples = [_make_sample() for _ in range(3)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=_DERIVED_E1_COLS,
        )
        for col in _DERIVED_E1_COLS:
            assert stats[col]["std"] > 0, f"{col}: std must be positive"


class TestComputeStatsDerivedNumerics:
    """Unit: numeric correctness for each derived feature."""

    def test_g_sin_gamma_mean(self) -> None:
        """mean(fdm_g_sin_gamma_ms2) == G * sin(gamma_known)."""
        gamma = 0.1  # rad
        samples = [_make_sample(gamma_rad=gamma)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_g_sin_gamma_ms2"],
        )
        expected = G * math.sin(gamma)
        assert stats["fdm_g_sin_gamma_ms2"]["mean"] == pytest.approx(expected, rel=1e-4)

    def test_cos_gamma_mean(self) -> None:
        """mean(fdm_cos_gamma) == cos(gamma_known)."""
        gamma = 0.05  # rad ≈ 2.9°
        samples = [_make_sample(gamma_rad=gamma)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_cos_gamma"],
        )
        expected = math.cos(gamma)
        assert stats["fdm_cos_gamma"]["mean"] == pytest.approx(expected, rel=1e-4)

    def test_g_over_v_mean(self) -> None:
        """mean(fdm_g_over_v) == G / tas for normal TAS."""
        tas = 200.0
        samples = [_make_sample(tas_ms=tas)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_g_over_v"],
        )
        expected = G / tas
        assert stats["fdm_g_over_v"]["mean"] == pytest.approx(expected, rel=1e-4)

    def test_q_sea_level_v200(self) -> None:
        """Sea level, V=200 m/s → q ≈ 24500 Pa (ISA rho0 ≈ 1.225 kg/m³)."""
        samples = [_make_sample(alt_m=0.0, tas_ms=200.0, temp_k=288.15)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_q_pa"],
        )
        q_mean = stats["fdm_q_pa"]["mean"]
        # Expected: 0.5 * 1.225 * 200^2 ≈ 24500 Pa
        assert 23000 <= q_mean <= 26000, f"q={q_mean:.0f} Pa not in [23000, 26000]"

    def test_q_range_realistic(self) -> None:
        """q at cruise (FL350, V=240 m/s) is in plausible range 8000-25000 Pa."""
        samples = [_make_sample(alt_m=10_500.0, tas_ms=240.0, temp_k=218.0)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_q_pa"],
        )
        q_mean = stats["fdm_q_pa"]["mean"]
        assert 8000 <= q_mean <= 30000, f"q={q_mean:.0f} Pa out of range"


class TestComputeStatsDerivedEdgeCases:
    """Unit: edge cases for derived feature computation."""

    def test_q_isa_fallback_no_temp_col(self) -> None:
        """When e_cols has no era_temp_K, q falls back to ISA — no crash."""
        e_cols_no_temp = ["fdm_long_wind_ms"]  # no era_temp_K
        samples = [
            FlightSample(
                x=torch.tensor([[0.0, 0.05, 200.0]] * 20, dtype=torch.float32),
                u=torch.zeros(20, 0),
                e=torch.zeros(20, 1),  # only wind column
                dx=torch.zeros(20, 3),
                e1=None,
            )
        ]
        # Should not raise
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            e_cols_no_temp,
            _DX_COLS,
            e1_cols=["fdm_q_pa"],
        )
        assert "fdm_q_pa" in stats
        q_mean = stats["fdm_q_pa"]["mean"]
        # ISA at sea level with V=200 → ~24500 Pa
        assert 23000 <= q_mean <= 26000, f"q_isa={q_mean:.0f} Pa not in range"

    def test_g_over_v_clamp_at_low_tas(self) -> None:
        """TAS=0.5 → clamped to 1.0 → G/1.0 = 9.80665."""
        samples = [_make_sample(tas_ms=0.5)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_g_over_v"],
        )
        expected = G / 1.0  # clamp floor is 1.0
        assert stats["fdm_g_over_v"]["mean"] == pytest.approx(expected, rel=1e-4)

    def test_existing_e1_tensor_takes_precedence(self) -> None:
        """If s.e1 already contains a derived column, tensor path wins.

        In production, e1 columns come from the parquet (empty/zero for
        derived), but this guards against accidental overwrite.
        """
        # Build samples WITH e1 tensor at index 0 = fdm_g_sin_gamma_ms2
        # Set the tensor to a known constant (99.0) — different from analytic value.
        e1_cols_subset = ["fdm_g_sin_gamma_ms2"]
        samples = [
            FlightSample(
                x=torch.tensor([[0.0, 0.0, 200.0]] * 20, dtype=torch.float32),
                u=torch.zeros(20, 0),
                e=torch.zeros(20, 2),
                dx=torch.zeros(20, 3),
                e1=torch.full((20, 1), 99.0),  # synthetic value
            )
            for _ in range(3)
        ]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=e1_cols_subset,
        )
        # Tensor path must take precedence: mean should be ~99.0 not G*sin(0)=0
        assert stats["fdm_g_sin_gamma_ms2"]["mean"] == pytest.approx(99.0, abs=1e-4)

    def test_unknown_derived_col_skipped(self) -> None:
        """A column not in DERIVED_FEATURES and not in s.e1 is silently skipped."""
        samples = [_make_sample()]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_unknown_col_xyz"],
        )
        assert "fdm_unknown_col_xyz" not in stats


class TestComputeStatsDerivedAnalyticAccuracy:
    """Unit: verify analytic formulas match numpy reference implementation."""

    def test_q_formula_matches_reference(self) -> None:
        """q formula matches manual reference: 0.5 * (p / (R*T)) * V^2."""
        alt = 8000.0
        tas = 230.0
        temp = 236.0
        samples = [_make_sample(alt_m=alt, tas_ms=tas, temp_k=temp)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_q_pa"],
        )
        p = float(np.asarray(isa_pressure(np.array([alt])))[0])
        rho_ref = p / (R * temp)
        q_ref = 0.5 * rho_ref * tas**2
        assert stats["fdm_q_pa"]["mean"] == pytest.approx(q_ref, rel=1e-3)

    def test_g_over_v_formula_matches_reference(self) -> None:
        """g_over_v formula matches G / tas for tas > 1.0."""
        tas = 180.0
        samples = [_make_sample(tas_ms=tas)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            e1_cols=["fdm_g_over_v"],
        )
        assert stats["fdm_g_over_v"]["mean"] == pytest.approx(G / tas, rel=1e-4)
