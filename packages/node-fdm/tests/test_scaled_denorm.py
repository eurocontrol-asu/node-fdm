"""Tests for the 'scaled' denormalization mode using p99.9 + dx_bounds cap.

Covers: compute_stats p999 stat, OutputDenormalizer scaled mode,
StructuredLayer passthrough, FDM integration, and edge cases.
"""

from __future__ import annotations

import pytest
import torch

from node_fdm.dataset import FlightSample, compute_stats
from node_fdm.layers.normalizers import OutputDenormalizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(
    n: int = 200,
    seq_len: int = 50,
    n_x: int = 3,
    n_u: int = 2,
    n_e: int = 2,
    n_dx: int = 2,
) -> list[FlightSample]:
    """Create deterministic samples with a known distribution."""
    torch.manual_seed(42)
    return [
        FlightSample(
            x=torch.randn(seq_len, n_x),
            u=torch.randn(seq_len, n_u),
            e=torch.randn(seq_len, n_e),
            dx=torch.randn(seq_len, n_dx),
        )
        for _ in range(n)
    ]


# ===================================================================
# Unit tests
# ===================================================================


class TestComputeStatsP999:
    """compute_stats must include a p999 (p99.9) key per column."""

    def test_p999_key_present(self) -> None:
        """Each column in stats has a 'p999' entry."""
        samples = _make_samples(n=50)
        x_cols = ["x1", "x2", "x3"]
        u_cols = ["u1", "u2"]
        e_cols = ["e1", "e2"]
        dx_cols = ["dx1", "dx2"]

        stats = compute_stats(samples, x_cols, u_cols, e_cols, dx_cols)

        for col in x_cols + u_cols + e_cols + dx_cols:
            assert "p999" in stats[col], f"p999 missing for column {col}"

    def test_p999_correct_value(self) -> None:
        """p999 matches the 99.9th percentile of absolute values."""
        # Use known data: 1000 linearly spaced values from -1 to 1
        n_pts = 1000
        data = torch.linspace(-1.0, 1.0, n_pts).unsqueeze(1)
        samples = [
            FlightSample(
                x=data,
                u=data[:, :0].expand(n_pts, 0),  # 0 u_cols
                e=data[:, :0].expand(n_pts, 0),  # 0 e_cols
                dx=data,
            )
        ]
        x_cols = ["x1"]
        u_cols: list[str] = []
        e_cols: list[str] = []
        dx_cols = ["dx1"]

        stats = compute_stats(samples, x_cols, u_cols, e_cols, dx_cols)

        # p99.9 of |linspace(-1,1,1000)| ≈ 1.0 (the 999th value out of 1000)
        expected = torch.quantile(data[:, 0].abs(), 0.999).item()
        assert abs(stats["x1"]["p999"] - expected) < 1e-4


class TestDenormScaledMode:
    """OutputDenormalizer 'scaled' mode: z * scale, clamped to ±cap."""

    def test_linear_scaling(self) -> None:
        """z=1.5 with scale=0.01 → 0.015 (below cap)."""
        denorm = OutputDenormalizer(
            mean_dict={"d_gamma": 0.0},
            std_dict={"d_gamma": 0.001},
            max_dict={"d_gamma": 0.02},
            modes={"d_gamma": "scaled"},
            scale_dict={"d_gamma": 0.01},
            cap_dict={"d_gamma": 0.03},
        )
        out = denorm(torch.tensor([1.5]), "d_gamma")
        assert torch.isclose(out, torch.tensor([0.015])).all()

    def test_capped(self) -> None:
        """z=5.0 with scale=0.01, cap=0.03 → 0.03 (capped)."""
        denorm = OutputDenormalizer(
            mean_dict={"d_gamma": 0.0},
            std_dict={"d_gamma": 0.001},
            max_dict={"d_gamma": 0.02},
            modes={"d_gamma": "scaled"},
            scale_dict={"d_gamma": 0.01},
            cap_dict={"d_gamma": 0.03},
        )
        out = denorm(torch.tensor([5.0]), "d_gamma")
        assert torch.isclose(out, torch.tensor([0.03])).all()

    def test_negative_capped(self) -> None:
        """Negative z is clamped symmetrically."""
        denorm = OutputDenormalizer(
            mean_dict={"d_gamma": 0.0},
            std_dict={"d_gamma": 0.001},
            max_dict={"d_gamma": 0.02},
            modes={"d_gamma": "scaled"},
            scale_dict={"d_gamma": 0.01},
            cap_dict={"d_gamma": 0.03},
        )
        out = denorm(torch.tensor([-5.0]), "d_gamma")
        assert torch.isclose(out, torch.tensor([-0.03])).all()


class TestDenormScaledGradient:
    """Gradients through 'scaled' mode must be finite and non-zero."""

    def test_gradient_flows(self) -> None:
        """Forward + backward through scaled mode produces usable gradients."""
        denorm = OutputDenormalizer(
            mean_dict={"d_gamma": 0.0},
            std_dict={"d_gamma": 0.001},
            max_dict={"d_gamma": 0.02},
            modes={"d_gamma": "scaled"},
            scale_dict={"d_gamma": 0.01},
            cap_dict={"d_gamma": 0.03},
        )
        x = torch.tensor([1.5], requires_grad=True)
        out = denorm(x, "d_gamma")
        out.backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()
        assert (x.grad != 0).all()


class TestDenormNormalClampUnchanged:
    """Existing 'normal_clamp' mode must be unaffected by scaled additions."""

    def test_normal_clamp_behavior(self) -> None:
        """normal_clamp: mean + z*std, clamped to ±max_ratio*max."""
        denorm = OutputDenormalizer(
            mean_dict={"speed": 200.0},
            std_dict={"speed": 50.0},
            max_dict={"speed": 300.0},
            modes={"speed": "normal_clamp"},
        )
        # z=0 → mean = 200.0
        out = denorm(torch.tensor([0.0]), "speed")
        assert torch.isclose(out, torch.tensor([200.0])).all()

        # z=1 → 200 + 50 = 250, within clamp
        out = denorm(torch.tensor([1.0]), "speed")
        assert torch.isclose(out, torch.tensor([250.0])).all()

        # z=10 → 200 + 500 = 700, clamped to 1.2*300 = 360
        out = denorm(torch.tensor([10.0]), "speed")
        assert torch.isclose(out, torch.tensor([360.0])).all()


class TestStructuredLayerScaledPassthrough:
    """StructuredLayer must forward scale/cap dicts to OutputDenormalizer."""

    def test_denormalizer_receives_scaled_config(self) -> None:
        """StructuredLayer with denormalize_modes passes scale/cap to denormalizer."""
        from node_fdm.layers.structured import StructuredLayer

        input_cols = ["x1", "x2"]
        output_cols = ["d_gamma", "d_tas"]
        input_mean = {"x1": 0.0, "x2": 0.0}
        input_std = {"x1": 1.0, "x2": 1.0}
        output_mean = {"d_gamma": 0.0, "d_tas": 0.0}
        output_std = {"d_gamma": 0.001, "d_tas": 0.5}
        output_max = {"d_gamma": 0.02, "d_tas": 5.0}

        layer = StructuredLayer(
            input_cols=input_cols,
            input_stats=(input_mean, input_std),
            output_cols=output_cols,
            output_stats=(output_mean, output_std, output_max),
            backbone_dim=16,
            backbone_depth=1,
            head_dim=8,
            head_depth=1,
            denormalize_modes={"d_gamma": "scaled", "d_tas": "normal_clamp"},
            scale_dict={"d_gamma": 0.01},
            cap_dict={"d_gamma": 0.03},
        )

        # Verify the denormalizer has the correct mode for d_gamma
        assert layer.denormalizer._modes["d_gamma"] == "scaled"
        assert layer.denormalizer._modes["d_tas"] == "normal_clamp"


# ===================================================================
# Functional tests
# ===================================================================


class TestFdmAdsbScaledDenorm:
    """Full FDM with node_adsb_v1 and scaled denorm covers wider d_gamma range."""

    def test_d_gamma_output_range(self) -> None:
        """d_gamma output range covers [-0.03, 0.03], not limited to [-0.003, 0.003]."""
        from node_fdm.architectures.adsb import NODE_ADSB_V1
        from node_fdm.models.fdm import FlightDynamicsModel

        # Build stats dict with p999 for all columns used by the architecture
        dx_col_names = [c for _, c in NODE_ADSB_V1.dx_cols]
        all_cols = (
            NODE_ADSB_V1.x_cols
            + NODE_ADSB_V1.u_cols
            + NODE_ADSB_V1.e0_cols
            + NODE_ADSB_V1.e1_cols
            + dx_col_names
        )
        stats_dict: dict[str, dict[str, float]] = {}
        for col in all_cols:
            stats_dict[col] = {"mean": 0.0, "std": 1.0, "max": 3.0, "p999": 2.5}

        # Override d_gamma stats to match realistic flight data
        stats_dict["fdm_d_gamma_rads"] = {
            "mean": 0.0,
            "std": 0.00116,
            "max": 0.025,
            "p999": 0.0087,
        }

        model = FlightDynamicsModel(
            spec=NODE_ADSB_V1,
            stats_dict=stats_dict,
            model_params=(1, 1, 16),  # small model for test speed
        )

        # Forward pass with extreme inputs to push d_gamma to its limits
        batch = 64
        n_x = len(NODE_ADSB_V1.x_cols)
        n_u = len(NODE_ADSB_V1.u_cols)
        n_e = len(NODE_ADSB_V1.e0_cols)

        torch.manual_seed(0)
        x = torch.randn(batch, n_x) * 3.0
        u = torch.randn(batch, n_u) * 3.0
        e = torch.randn(batch, n_e) * 3.0

        with torch.no_grad():
            dx = model(x, u, e)

        # Extract d_gamma column (dx_cols is list[tuple[int, str]])
        dx_col_names = [name for _, name in NODE_ADSB_V1.dx_cols]
        d_gamma_idx = dx_col_names.index("fdm_d_gamma_rads")
        d_gamma = dx[:, d_gamma_idx]

        # With scaled mode (p999=0.0087), the network can produce values
        # up to ±0.03 (the dx_bounds cap). With old std=0.00116, max would
        # be ~0.003. Check that the achievable range exceeds the old limit.
        max_abs = d_gamma.abs().max().item()
        assert max_abs > 0.003, f"d_gamma max |{max_abs:.6f}| should exceed 0.003 with scaled mode"


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCaseNoModes:
    """No denormalize_modes in config → all columns default to normal_clamp."""

    def test_default_modes(self) -> None:
        """When modes is None, all columns use normal_clamp."""
        denorm = OutputDenormalizer(
            mean_dict={"a": 1.0, "b": 2.0},
            std_dict={"a": 0.5, "b": 1.0},
            max_dict={"a": 3.0, "b": 5.0},
        )
        assert denorm._modes == {"a": "normal_clamp", "b": "normal_clamp"}


class TestEdgeCaseMixedModes:
    """Mixed modes: each column uses its own denormalization mode."""

    def test_mixed_scaled_and_normal_clamp(self) -> None:
        """d_gamma uses 'scaled', d_tas uses 'normal_clamp'."""
        denorm = OutputDenormalizer(
            mean_dict={"d_gamma": 0.0, "d_tas": 100.0},
            std_dict={"d_gamma": 0.001, "d_tas": 10.0},
            max_dict={"d_gamma": 0.02, "d_tas": 200.0},
            modes={"d_gamma": "scaled", "d_tas": "normal_clamp"},
            scale_dict={"d_gamma": 0.01},
            cap_dict={"d_gamma": 0.03},
        )
        # d_gamma: scaled → z*scale
        out_gamma = denorm(torch.tensor([1.0]), "d_gamma")
        assert torch.isclose(out_gamma, torch.tensor([0.01])).all()

        # d_tas: normal_clamp → mean + z*std
        out_tas = denorm(torch.tensor([1.0]), "d_tas")
        assert torch.isclose(out_tas, torch.tensor([110.0])).all()


class TestEdgeCaseP999Missing:
    """Old stats dict without p999 key → fallback or clear error."""

    def test_missing_p999_raises(self) -> None:
        """FDM with scaled mode but no p999 in stats raises clear error."""
        from node_fdm.architectures.adsb import NODE_ADSB_V1
        from node_fdm.models.fdm import FlightDynamicsModel

        # Stats without p999
        dx_col_names = [c for _, c in NODE_ADSB_V1.dx_cols]
        all_cols = (
            NODE_ADSB_V1.x_cols
            + NODE_ADSB_V1.u_cols
            + NODE_ADSB_V1.e0_cols
            + NODE_ADSB_V1.e1_cols
            + dx_col_names
        )
        stats_dict: dict[str, dict[str, float]] = {}
        for col in all_cols:
            stats_dict[col] = {"mean": 0.0, "std": 1.0, "max": 3.0}
            # No "p999" key

        with pytest.raises((KeyError, ValueError)):
            FlightDynamicsModel(
                spec=NODE_ADSB_V1,
                stats_dict=stats_dict,
                model_params=(1, 1, 16),
            )
