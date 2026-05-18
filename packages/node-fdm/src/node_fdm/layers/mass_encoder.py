"""Monotone linear mass encoder layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from node_fdm.layers.normalizers import InputNormalizer

__all__ = ["MassEncoderLinear"]


class MassEncoderLinear(nn.Module):
    """Linear monotone encoder mapping flight features to initial mass."""

    signs: torch.Tensor
    oew_kg: torch.Tensor
    mtow_kg: torch.Tensor

    def __init__(
        self,
        feature_stats: dict[str, dict[str, float]],
        feature_cols: list[str],
        expected_signs: list[float],
        oew_kg: float,
        mtow_kg: float,
        b0_init: float = 0.85,
    ) -> None:
        """Initialize the monotone mass encoder."""
        super().__init__()
        if len(feature_cols) != len(expected_signs):
            msg = "feature_cols and expected_signs must have the same length"
            raise ValueError(msg)

        self.feature_cols = list(feature_cols)
        mean_dict = {col: feature_stats[col]["mean"] for col in self.feature_cols}
        std_dict = {col: feature_stats[col]["std"] for col in self.feature_cols}
        modes: dict[str, str | None] = dict.fromkeys(self.feature_cols, "normal")
        self.normalizer = InputNormalizer(mean_dict, std_dict, modes)

        self.b_raw = nn.Parameter(torch.zeros(len(self.feature_cols), dtype=torch.float32))
        self.b0 = nn.Parameter(torch.tensor([b0_init], dtype=torch.float32))

        self.register_buffer("signs", torch.tensor(expected_signs, dtype=torch.float32))
        self.register_buffer("oew_kg", torch.tensor(oew_kg, dtype=torch.float32))
        self.register_buffer("mtow_kg", torch.tensor(mtow_kg, dtype=torch.float32))

    def forward(self, flight_features: torch.Tensor) -> torch.Tensor:
        """Encode batched flight features into initial mass estimates."""
        z_norm = torch.stack(
            [
                self.normalizer(flight_features[..., idx], col)
                for idx, col in enumerate(self.feature_cols)
            ],
            dim=-1,
        )
        b = self.signs * functional.softplus(self.b_raw)
        z = self.b0 + (z_norm * b).sum(dim=-1)
        alpha = torch.sigmoid(z)
        oew_kg = self.oew_kg.double()
        mtow_kg = self.mtow_kg.double()
        return oew_kg + alpha.double() * (mtow_kg - oew_kg)

    def effective_coefficients(self) -> dict[str, float]:
        """Return signed effective coefficients for diagnostics."""
        with torch.no_grad():
            b = self.signs * functional.softplus(self.b_raw)
            coefficients = {"b0": float(self.b0.item())}
            coefficients.update(
                {col: float(value.item()) for col, value in zip(self.feature_cols, b, strict=True)}
            )
        return coefficients
