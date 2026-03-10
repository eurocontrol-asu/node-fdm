"""Engine layer for QAR-based flight dynamics models.

Extends ``StructuredLayer`` with engine-specific denormalization for
N1 (percentage scale) and fuel flow.
"""

from __future__ import annotations

import torch

from node_fdm.layers.structured import StructuredLayer

__all__ = [
    "EngineLayer",
]

_N1_MAX: int = 100


class EngineLayer(StructuredLayer):
    """Engine-specific structured layer for N1 and fuel flow predictions.

    N1 columns are scaled to the [0, ``_N1_MAX``] range. Other outputs
    use the standard denormalizer.  "N1" columns are detected by checking
    if their name contains ``"N1"`` (case-insensitive).
    """

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Produce denormalized engine predictions.

        Args:
            x_dict: Mapping of column names to input tensors.

        Returns:
            Dictionary with engine output predictions keyed by column name.
        """
        out_norm_dict = self.forward_trunk_head(x_dict)
        out_pred: dict[str, torch.Tensor] = {}

        for col in self.output_cols:
            norm_val = out_norm_dict[col].squeeze(-1)
            if "N1" in col.upper():
                # N1: scale normalized output to percentage
                out_pred[col] = _N1_MAX * norm_val
            else:
                # Fuel flow or other: standard denormalization
                out_pred[col] = self.denormalizer(norm_val, col)

        return out_pred
