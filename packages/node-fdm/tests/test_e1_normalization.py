"""Tests for AXM-843: E1 columns must be z-score normalized in StructuredLayer inputs.

Bug: model_stats passed to FlightDynamicsModel is computed WITHOUT e1_cols,
so E1 columns in StructuredLayer.input_cols have no normalization stats and
pass through unnormalized — breaking the O(1) input assumption.

Validates that:
- model_stats includes E1 column statistics after the fix.
- Overlapping columns (e.g. fdm_d_vz_ms in both DX and E1) keep DX stats.
- StructuredLayer normalizes ALL inputs (including E1) to O(1) magnitude.
"""

from __future__ import annotations

import torch

from node_fdm.architectures.registry import get
from node_fdm.dataset import FlightDataset, FlightSample, compute_stats
from node_fdm.layers.structured import StructuredLayer
from node_fdm.trainer import ODETrainer, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = get("node_adsb_v1")
_DX_COL_NAMES = [col for _, col in _SPEC.dx_cols]
_ODE_LAYER = next(ly for ly in _SPEC.layers if ly.trainable)
_U_ODE_COLS = [c for c in _SPEC.u_cols if c in _ODE_LAYER.input_cols]


def _make_samples(n: int = 20, seq_len: int = 10) -> list[FlightSample]:
    """Build synthetic FlightSamples matching node_adsb_v1 column counts."""
    return [
        FlightSample(
            x=torch.randn(seq_len, len(_SPEC.x_cols)),
            u=torch.randn(seq_len, len(_SPEC.u_cols)),
            e=torch.randn(seq_len, len(_SPEC.e0_cols)),
            dx=torch.randn(seq_len, len(_DX_COL_NAMES)),
            e1=torch.randn(seq_len, len(_SPEC.e1_cols)),
        )
        for _ in range(n)
    ]


def _make_stats(cols: list[str]) -> dict[str, dict[str, float]]:
    """Build dummy stats covering all requested columns."""
    return {col: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 0.8} for col in cols}


# ===========================================================================
# Test 1 — model_stats includes E1 column statistics
# ===========================================================================


class TestModelStatsIncludeE1Columns:
    """After fix, FlightDynamicsModel.stats_dict must contain E1 column keys."""

    def test_model_stats_include_e1_columns(self, tmp_path: object) -> None:
        """Build trainer with node_adsb_v1 arch, inspect model_stats keys.

        All E1 input columns that appear in the trainable layer's input_cols
        must have entries in model.stats_dict.
        """
        samples = _make_samples()
        ds = FlightDataset(samples)
        config = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_e1_norm",
            seq_len=10,
            shift=10,
            epochs=1,
        )
        trainer = ODETrainer(
            config=config,
            train_dataset=ds,
            val_dataset=ds,
            model_dir=tmp_path,  # type: ignore[arg-type]
        )

        model_stats_keys = set(trainer.model.stats_dict.keys())
        e1_in_layer = [c for c in _SPEC.e1_cols if c in _ODE_LAYER.input_cols]

        for col in e1_in_layer:
            assert col in model_stats_keys, (
                f"E1 column '{col}' missing from model_stats — "
                f"StructuredLayer input will not be z-score normalized"
            )


# ===========================================================================
# Test 2 — E1 stats do not overwrite DX stats
# ===========================================================================


class TestE1StatsNoOverwriteDx:
    """When a column appears in both DX and E1, DX stats must be preserved."""

    def test_e1_stats_no_overwrite_dx(self) -> None:
        """fdm_d_vz_ms is in both dx_cols and e1_cols.

        compute_stats with e1_cols must keep the DX-derived statistics
        (computed first from the main data tensor), not overwrite them
        with E1-derived values.
        """
        seq_len = 50
        n_samples = 10

        # Use distinct distributions so DX and E1 stats are distinguishable
        dx_value = 42.0  # constant → mean=42, std≈1e-6
        e1_value = -99.0  # constant → mean=-99

        samples = [
            FlightSample(
                x=torch.randn(seq_len, len(_SPEC.x_cols)),
                u=(
                    torch.randn(seq_len, len(_SPEC.u_cols))
                    if _U_ODE_COLS
                    else torch.empty(seq_len, 0)
                ),
                e=torch.randn(seq_len, len(_SPEC.e0_cols)),
                dx=torch.full((seq_len, len(_DX_COL_NAMES)), dx_value),
                e1=torch.full((seq_len, len(_SPEC.e1_cols)), e1_value),
            )
            for _ in range(n_samples)
        ]

        stats = compute_stats(
            samples,
            x_cols=_SPEC.x_cols,
            u_cols=_U_ODE_COLS,
            e_cols=_SPEC.e0_cols,
            dx_cols=_DX_COL_NAMES,
            e1_cols=_SPEC.e1_cols,
        )

        overlap_col = "fdm_d_vz_ms"
        assert overlap_col in _DX_COL_NAMES, "Test assumption: fdm_d_vz_ms in dx_cols"
        assert overlap_col in _SPEC.e1_cols, "Test assumption: fdm_d_vz_ms in e1_cols"

        # Stats must reflect the DX distribution, not E1
        assert abs(stats[overlap_col]["mean"] - dx_value) < 1.0, (
            f"Overlap column '{overlap_col}' mean={stats[overlap_col]['mean']:.2f} "
            f"does not match DX value {dx_value} — E1 stats overwrote DX"
        )


# ===========================================================================
# Test 3 — StructuredLayer normalizes ALL inputs (including E1)
# ===========================================================================


class TestStructuredLayerAllInputsNormalized:
    """StructuredLayer with merged stats must normalize every input to O(1)."""

    def test_structured_layer_all_inputs_normalized(self) -> None:
        """Build StructuredLayer with stats covering all input_cols (incl. E1).

        Pass tensors with large-magnitude values and verify that after
        the normalizer, all inputs are O(1).
        """
        input_cols = _ODE_LAYER.input_cols
        output_cols = _ODE_LAYER.output_cols

        # Large-magnitude inputs: mean=1000, std=200
        means = dict.fromkeys(input_cols, 1000.0)
        stds = dict.fromkeys(input_cols, 200.0)

        # Output stats (needed by constructor, not tested here)
        out_means = dict.fromkeys(output_cols, 0.0)
        out_stds = dict.fromkeys(output_cols, 1.0)
        out_maxs = dict.fromkeys(output_cols, 1.0)

        layer = StructuredLayer(
            input_cols=input_cols,
            input_stats=(means, stds),
            output_cols=output_cols,
            output_stats=(out_means, out_stds, out_maxs),
            backbone_dim=16,
            backbone_depth=1,
            head_dim=8,
            head_depth=1,
        )

        # Feed tensors drawn from N(1000, 200) — without normalization
        # the backbone would see inputs of magnitude ~1000
        batch = 8
        x_dict = {col: torch.randn(batch, 1) * 200 + 1000 for col in input_cols}

        # Normalize each input through the layer's normalizer
        normalizer = layer.normalizer
        for col in input_cols:
            normed = normalizer(x_dict[col], col)
            assert normed.abs().mean().item() < 10.0, (
                f"Column '{col}' not normalized — "
                f"mean abs = {normed.abs().mean().item():.1f} (expected O(1))"
            )
