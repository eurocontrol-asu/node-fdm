"""Tests for NN-output derived stats and the spec-level ``nn_output_caps``.

Covers the data-driven replacement of the legacy hardcoded
``scale_overrides`` / ``cap_overrides`` for ADS-N NN heads:

* The 3 inverse-PhysicsLayer entries in ``DERIVED_FEATURES``.
* ``compute_stats(..., derived_cols=...)`` emits ``mean/std/max/p999`` for
  columns that never appear in any tensor.
* ``ArchitectureSpec.nn_output_caps`` overrides the data-driven p999 in
  ``_create_structured_layer`` (training and prod paths).
* ``NODE_ADSB_V1`` no longer carries ``scale_overrides`` /
  ``cap_overrides`` and instead lists ``derived_output_cols`` and
  ``nn_output_caps``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from node_fdm.architectures.adsb import NODE_ADSB_V1
from node_fdm.dataset import DERIVED_FEATURES, FlightSample, compute_stats
from node_fdm.layers.physics import V_MIN_CLAMP, G

# ── synthetic-sample columns matching the inverse-PhysicsLayer inputs ──────
_X_COLS = ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]
_U_COLS: list[str] = []
_E_COLS = ["fdm_long_wind_ms", "era_temp_K"]
_DX_COLS = ["fdm_d_alt_ms", "fdm_d_gamma_rads", "fdm_d_tas_ms2", "fdm_d_heading_rads"]


def _make_sample(
    *,
    alt_m: float = 5000.0,
    gamma_rad: float = 0.05,
    tas_ms: float = 200.0,
    d_tas: float = 0.3,
    d_gamma: float = 0.001,
    d_heading: float = 0.02,
    seq_len: int = 16,
) -> FlightSample:
    """Build a sample with constant per-step state and dx values."""
    x = torch.tensor([[alt_m, gamma_rad, tas_ms]] * seq_len, dtype=torch.float32)
    u = torch.zeros(seq_len, 0)
    e = torch.tensor([[0.0, 255.65]] * seq_len, dtype=torch.float32)
    dx = torch.tensor(
        [[0.0, d_gamma, d_tas, d_heading]] * seq_len,
        dtype=torch.float32,
    )
    return FlightSample(x=x, u=u, e=e, dx=dx, e1=None)


# ===========================================================================
# Unit — DERIVED_FEATURES registry
# ===========================================================================


class TestDerivedFeaturesRegistry:
    """The 3 NN-output inversion entries are present and callable."""

    def test_nn_output_keys_present(self) -> None:
        """All 3 inverse-PhysicsLayer keys are registered."""
        for col in ("fdm_a_spec_ms2", "fdm_n_z_residual", "fdm_phi_bank_rad"):
            assert col in DERIVED_FEATURES, f"{col} missing from DERIVED_FEATURES"

    def test_uniform_signature(self) -> None:
        """All registered computers accept the 6-arg uniform signature."""
        x = np.array([[1000.0, 0.05, 200.0]], dtype=np.float64)
        e = np.array([[0.0, 255.65]], dtype=np.float64)
        dx = np.array([[0.0, 0.001, 0.3, 0.02]], dtype=np.float64)
        for col, fn in DERIVED_FEATURES.items():
            out = fn(x, e, dx, _X_COLS, _E_COLS, _DX_COLS)
            assert out.shape == (1,), f"{col} returned shape {out.shape}"
            assert np.isfinite(out).all(), f"{col} produced non-finite value"


# ===========================================================================
# Unit — analytic correctness of the inversion formulas
# ===========================================================================


class TestInversePhysicsLayerNumerics:
    """Each inverse must algebraically match the forward PhysicsLayer."""

    def test_a_spec_inverse(self) -> None:
        """``a_spec - g*sin(gamma)`` round-trips ``d_tas``."""
        gamma, d_tas = 0.05, 0.3
        sample = _make_sample(gamma_rad=gamma, d_tas=d_tas)
        stats = compute_stats(
            [sample],
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            derived_cols=["fdm_a_spec_ms2"],
        )
        expected = d_tas + G * math.sin(gamma)
        assert stats["fdm_a_spec_ms2"]["mean"] == pytest.approx(expected, abs=1e-5)

    def test_n_z_residual_inverse(self) -> None:
        """``(V/g)*d_gamma + cos(gamma) - 1`` matches the forward equation."""
        gamma, tas, d_gamma = 0.05, 200.0, 0.001
        sample = _make_sample(gamma_rad=gamma, tas_ms=tas, d_gamma=d_gamma)
        stats = compute_stats(
            [sample],
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            derived_cols=["fdm_n_z_residual"],
        )
        v_safe = max(tas, V_MIN_CLAMP)
        expected = (v_safe / G) * d_gamma + math.cos(gamma) - 1.0
        assert stats["fdm_n_z_residual"]["mean"] == pytest.approx(expected, abs=1e-5)

    def test_phi_bank_inverse(self) -> None:
        """``atan((V/g)*d_heading)`` matches the coordinated-turn equation."""
        tas, d_heading = 200.0, 0.02
        sample = _make_sample(tas_ms=tas, d_heading=d_heading)
        stats = compute_stats(
            [sample],
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            derived_cols=["fdm_phi_bank_rad"],
        )
        v_safe = max(tas, V_MIN_CLAMP)
        expected = math.atan((v_safe / G) * d_heading)
        assert stats["fdm_phi_bank_rad"]["mean"] == pytest.approx(expected, abs=1e-5)

    def test_v_min_clamp_applied(self) -> None:
        """Below-stall TAS is floored to V_MIN_CLAMP for the 1/V term."""
        # tas=10 m/s would otherwise blow up phi_bank; clamp keeps it sane.
        sample = _make_sample(tas_ms=10.0, d_heading=0.02)
        stats = compute_stats(
            [sample],
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            derived_cols=["fdm_phi_bank_rad"],
        )
        expected = math.atan((V_MIN_CLAMP / G) * 0.02)
        assert stats["fdm_phi_bank_rad"]["mean"] == pytest.approx(expected, abs=1e-5)


# ===========================================================================
# Unit — compute_stats(..., derived_cols=...) output shape
# ===========================================================================


class TestComputeStatsDerivedCols:
    """``derived_cols`` produces full {mean, std, max, p999} dicts."""

    def test_all_keys_present(self) -> None:
        """Each derived col has the 4 expected stat keys."""
        samples = [_make_sample(d_tas=0.1 * i) for i in range(20)]
        stats = compute_stats(
            samples,
            _X_COLS,
            _U_COLS,
            _E_COLS,
            _DX_COLS,
            derived_cols=["fdm_a_spec_ms2", "fdm_n_z_residual", "fdm_phi_bank_rad"],
        )
        for col in ("fdm_a_spec_ms2", "fdm_n_z_residual", "fdm_phi_bank_rad"):
            assert set(stats[col].keys()) >= {"mean", "std", "max", "p999"}
            assert math.isfinite(stats[col]["p999"])

    def test_unknown_derived_col_raises(self) -> None:
        """An unregistered derived column name raises KeyError."""
        with pytest.raises(KeyError, match="DERIVED_FEATURES"):
            compute_stats(
                [_make_sample()],
                _X_COLS,
                _U_COLS,
                _E_COLS,
                _DX_COLS,
                derived_cols=["fdm_does_not_exist"],
            )

    def test_dx_branch_still_emits_p999(self) -> None:
        """Regression: dx_cols still receive p999 (untouched main path)."""
        samples = [_make_sample(d_tas=0.1 * i) for i in range(10)]
        stats = compute_stats(samples, _X_COLS, _U_COLS, _E_COLS, _DX_COLS)
        for col in _DX_COLS:
            assert "p999" in stats[col]


# ===========================================================================
# Unit — ArchitectureSpec extension and ADSB refactor
# ===========================================================================


class TestArchitectureSpecExtension:
    """New optional fields default cleanly and accept overrides."""

    def test_defaults_empty(self) -> None:
        """Specs that omit the new fields get empty defaults (backward compat)."""
        # qar arch (loaded for side effect of registry import) is the
        # canonical "no overrides" baseline.
        from node_fdm.architectures import qar as _qar  # noqa: F401
        from node_fdm.architectures.registry import REGISTRY

        spec = REGISTRY["qar"]
        assert spec.derived_output_cols == []
        assert spec.nn_output_caps == {}

    def test_adsb_lists_three_derived_cols(self) -> None:
        """ADS-N declares the 3 NN-output cols for stats derivation."""
        assert set(NODE_ADSB_V1.derived_output_cols) == {
            "fdm_a_spec_ms2",
            "fdm_n_z_residual",
            "fdm_phi_bank_rad",
        }

    def test_adsb_no_legacy_overrides(self) -> None:
        """ADS-N layer configs no longer carry scale/cap overrides."""
        for layer in NODE_ADSB_V1.layers:
            assert "scale_overrides" not in layer.config, f"{layer.name} still has scale_overrides"
            assert "cap_overrides" not in layer.config, f"{layer.name} still has cap_overrides"


# ===========================================================================
# Unit — _create_structured_layer precedence (both training and prod paths)
# ===========================================================================


class _FakeStats:
    """Minimal stats_dict factory with the required keys."""

    @staticmethod
    def make(p999: float = 0.7) -> dict[str, dict[str, float]]:
        cols = (
            _X_COLS
            + _E_COLS
            + _DX_COLS
            + ["fdm_a_spec_ms2", "fdm_n_z_residual", "fdm_phi_bank_rad"]
        )
        return {
            col: {"mean": 0.0, "std": 1.0 + 1e-6, "max": 1.0, "p999": p999, "iqr": 1.0}
            for col in cols
        }


class TestCreateStructuredLayerPrecedence:
    """Scale is data-driven (p999); cap comes from nn_output_caps when set."""

    def _build_training_model(self) -> Any:
        """Helper: instantiate FlightDynamicsModel with a stats stub."""
        from node_fdm.models.fdm import FlightDynamicsModel

        return FlightDynamicsModel(
            spec=NODE_ADSB_V1,
            stats_dict=_FakeStats.make(p999=0.42),
            model_params=(1, 1, 4),
        )

    def test_scale_is_always_data_driven(self) -> None:
        """``scale`` reads from stats_dict[p999], never from nn_output_caps."""
        model = self._build_training_model()
        long_denorm = model.layers_dict["data_ode_long"].denormalizer
        lat_denorm = model.layers_dict["data_ode_lat"].denormalizer
        # All 3 NN outputs receive the stub p999 of 0.42 as scale, even
        # though phi_bank/a_spec/n_z_residual all have nn_output_caps.
        assert float(long_denorm.scale_fdm_a_spec_ms2) == pytest.approx(0.42)
        assert float(long_denorm.scale_fdm_n_z_residual) == pytest.approx(0.42)
        assert float(lat_denorm.scale_fdm_phi_bank_rad) == pytest.approx(0.42)

    def test_cap_uses_nn_output_caps_when_present(self) -> None:
        """``cap`` reads from spec.nn_output_caps for each declared col."""
        model = self._build_training_model()
        long_denorm = model.layers_dict["data_ode_long"].denormalizer
        lat_denorm = model.layers_dict["data_ode_lat"].denormalizer
        caps = NODE_ADSB_V1.nn_output_caps
        assert float(long_denorm.cap_fdm_a_spec_ms2) == pytest.approx(caps["fdm_a_spec_ms2"])
        assert float(long_denorm.cap_fdm_n_z_residual) == pytest.approx(caps["fdm_n_z_residual"])
        assert float(lat_denorm.cap_fdm_phi_bank_rad) == pytest.approx(caps["fdm_phi_bank_rad"])

    def test_cap_strictly_above_scale(self) -> None:
        """Sanity: cap > scale for every NN output (preserves tanh gradient)."""
        model = self._build_training_model()
        for layer_name in ("data_ode_long", "data_ode_lat"):
            denorm = model.layers_dict[layer_name].denormalizer
            for col in model.layers_dict[layer_name].output_cols:
                cap = float(getattr(denorm, f"cap_{col}"))
                scale = float(getattr(denorm, f"scale_{col}"))
                assert cap > scale, f"{col}: cap={cap} must exceed scale={scale}"
