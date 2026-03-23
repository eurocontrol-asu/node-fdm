"""Tests for AXM-759: replace alt_target with alt_diff as StructuredLayer control signal.

Unit tests validate the spec wiring (input_cols, u_cols).
Functional test validates a full forward pass with the new spec.
Edge-case test validates graceful degradation when alt_diff stats are missing.
"""

from __future__ import annotations

import torch

from node_fdm.architectures.registry import get
from node_fdm.models.fdm import FlightDynamicsModel

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestAdsbStructuredInputCols:
    """Verify StructuredLayer (layers[1]) input_cols after alt_target removal."""

    def test_adsb_structured_input_no_alt_target(self) -> None:
        """fdm_alt_target_m must NOT appear in the StructuredLayer input_cols."""
        spec = get("node_adsb_v1")
        structured_input_cols = spec.layers[1].input_cols
        assert "fdm_alt_target_m" not in structured_input_cols

    def test_adsb_structured_input_has_alt_diff(self) -> None:
        """fdm_alt_diff_m must appear in the StructuredLayer input_cols."""
        spec = get("node_adsb_v1")
        structured_input_cols = spec.layers[1].input_cols
        assert "fdm_alt_diff_m" in structured_input_cols

    def test_adsb_u_cols_has_alt_target(self) -> None:
        """fdm_alt_target_m must remain in the spec-level u_cols (TrajectoryLayer needs it)."""
        spec = get("node_adsb_v1")
        assert "fdm_alt_target_m" in spec.u_cols


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def _make_stats(cols: list[str]) -> dict[str, dict[str, float]]:
    """Build a dummy stats_dict covering all columns."""
    return {col: {"mean": 0.0, "std": 1.0, "max": 1.0} for col in cols}


class TestAdsbForwardPass:
    """Full forward pass with the modified spec."""

    def test_model_forward_without_alt_target_input(self) -> None:
        """Build FDM with new spec, run forward — output shape correct, no error."""
        spec = get("node_adsb_v1")
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        batch = 4
        x = torch.randn(batch, len(spec.x_cols))
        u = torch.randn(batch, len(spec.u_cols))
        e = torch.randn(batch, len(spec.e0_cols))

        out = model(x, u, e)
        assert out.shape == (batch, len(spec.dx_cols))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAdsbAltDiffEdgeCases:
    """Edge-case: missing stats for fdm_alt_diff_m."""

    def test_structured_layer_without_alt_diff_stats(self) -> None:
        """StructuredLayer builds even without alt_diff stats (degraded)."""
        spec = get("node_adsb_v1")
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols
        stats = _make_stats(all_cols)
        # Remove alt_diff stats to simulate AXM-758 not yet implemented
        stats.pop("fdm_alt_diff_m", None)

        # Model should build without error — alt_diff just won't be normalized
        model = FlightDynamicsModel(spec, stats)
        batch = 4
        x = torch.randn(batch, len(spec.x_cols))
        u = torch.randn(batch, len(spec.u_cols))
        e = torch.randn(batch, len(spec.e0_cols))

        out = model(x, u, e)
        assert out.shape == (batch, len(spec.dx_cols))
