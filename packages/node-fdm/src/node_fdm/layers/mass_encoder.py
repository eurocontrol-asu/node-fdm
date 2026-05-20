"""Monotone linear mass encoder layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from node_fdm.layers.normalizers import InputNormalizer

__all__ = ["MassEncoderLinear", "MassEncoderLinearTempered", "MassEncoderMLPMonotone"]


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


class MassEncoderLinearTempered(MassEncoderLinear):
    """Linear monotone encoder with a temperature-scaled sigmoid (Strategy C).

    Strict drop-in superset of :class:`MassEncoderLinear`. The pre-sigmoid
    activation ``z = b0 + Σ signs · softplus(b_raw) · z̃`` is divided by a
    fixed positive ``temperature`` before the sigmoid:

        alpha = sigmoid(z / T),  m_0 = OEW + alpha · (MTOW - OEW)

    With ``T > 1`` the sigmoid response is broadened, so a given linear
    drive places ``m_0`` closer to the operating-range middle rather than
    saturating to ``OEW`` / ``MTOW``. Diagnoses (and fixes) the bimodal
    saturation pathology flagged in Experiment 01.

    The parameter set is byte-identical to :class:`MassEncoderLinear`
    (``b_raw``, ``b0``) — only a non-trainable ``temperature`` buffer is
    added — so checkpoints across the two classes are interchangeable.
    """

    temperature: torch.Tensor

    def __init__(
        self,
        feature_stats: dict[str, dict[str, float]],
        feature_cols: list[str],
        expected_signs: list[float],
        oew_kg: float,
        mtow_kg: float,
        b0_init: float = 0.85,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            feature_stats=feature_stats,
            feature_cols=feature_cols,
            expected_signs=expected_signs,
            oew_kg=oew_kg,
            mtow_kg=mtow_kg,
            b0_init=b0_init,
        )
        if temperature <= 0.0:
            msg = f"temperature must be > 0, got {temperature}"
            raise ValueError(msg)
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, flight_features: torch.Tensor) -> torch.Tensor:
        z_norm = torch.stack(
            [
                self.normalizer(flight_features[..., idx], col)
                for idx, col in enumerate(self.feature_cols)
            ],
            dim=-1,
        )
        b = self.signs * functional.softplus(self.b_raw)
        z = self.b0 + (z_norm * b).sum(dim=-1)
        alpha = torch.sigmoid(z / self.temperature)
        oew_kg = self.oew_kg.double()
        mtow_kg = self.mtow_kg.double()
        return oew_kg + alpha.double() * (mtow_kg - oew_kg)


class MassEncoderMLPMonotone(nn.Module):
    """Strictly-monotone small MLP encoder (Strategy B).

    Architecture:
        * input layer: ``h0 = softplus(W1_raw) * signs · z̃ + b1``
          — each hidden unit is a *positive* linear combination of the
          expected-sign-projected normalised features, so monotonicity in
          each input feature is enforced.
        * hidden activation: ``softplus`` (monotone non-decreasing) —
          NOT SiLU, which has a minimum at ``x ≈ -1.28`` and would break
          monotonicity.
        * second layer: ``h1 = softplus(softplus(W2_raw) · h0 + b2)`` —
          positive weights via softplus keep monotonicity through the
          composition.
        * head: ``z = softplus(W3_raw) · h1 + b0`` (positive head weights,
          unconstrained bias).
        * output: ``alpha = sigmoid(z / T)`` then
          ``m = OEW + alpha · (MTOW - OEW)``.

    With ``softplus`` activations and positive intermediate weights, the
    composition is a monotone non-decreasing function of each
    expected-sign-projected feature — i.e. strictly monotone in each
    ``flight_features[i]`` along the sign listed in ``expected_signs``.

    Drops the strict linearity of :class:`MassEncoderLinear` so feature
    interactions can extend the prediction range beyond what a single
    weighted sum can produce.
    """

    signs: torch.Tensor
    oew_kg: torch.Tensor
    mtow_kg: torch.Tensor
    temperature: torch.Tensor

    def __init__(
        self,
        feature_stats: dict[str, dict[str, float]],
        feature_cols: list[str],
        expected_signs: list[float],
        oew_kg: float,
        mtow_kg: float,
        hidden_dim: int = 32,
        temperature: float = 3.0,
        b0_init: float = 0.85,
    ) -> None:
        super().__init__()
        if len(feature_cols) != len(expected_signs):
            msg = "feature_cols and expected_signs must have the same length"
            raise ValueError(msg)
        if temperature <= 0.0:
            msg = f"temperature must be > 0, got {temperature}"
            raise ValueError(msg)
        hidden_dim = int(hidden_dim)

        self.feature_cols = list(feature_cols)
        mean_dict = {col: feature_stats[col]["mean"] for col in self.feature_cols}
        std_dict = {col: feature_stats[col]["std"] for col in self.feature_cols}
        modes: dict[str, str | None] = dict.fromkeys(self.feature_cols, "normal")
        self.normalizer = InputNormalizer(mean_dict, std_dict, modes)

        n_in = len(self.feature_cols)
        # Scale each layer's raw weights so ``softplus(W_raw) ≈ 1 / fan_in``
        # at initialization — keeps the pre-sigmoid sum O(1) instead of
        # blowing up through positive-weight cascades and saturating the
        # output to MTOW before any gradient step. Random ``torch.randn x
        # 0.1`` adds symmetry-breaking noise on top of the deterministic
        # ``-log(fan_in)`` shift.
        w1_shift = -math.log(max(n_in, 1))
        w2_shift = -math.log(max(hidden_dim, 1))
        self.W1_raw = nn.Parameter(
            w1_shift + torch.randn(hidden_dim, n_in, dtype=torch.float32) * 0.1
        )
        self.b1 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        self.W2_raw = nn.Parameter(
            w2_shift + torch.randn(hidden_dim, hidden_dim, dtype=torch.float32) * 0.1
        )
        self.b2 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        self.W3_raw = nn.Parameter(w2_shift + torch.randn(hidden_dim, dtype=torch.float32) * 0.1)
        self.b0 = nn.Parameter(torch.tensor([b0_init], dtype=torch.float32))

        self.register_buffer("signs", torch.tensor(expected_signs, dtype=torch.float32))
        self.register_buffer("oew_kg", torch.tensor(oew_kg, dtype=torch.float32))
        self.register_buffer("mtow_kg", torch.tensor(mtow_kg, dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, flight_features: torch.Tensor) -> torch.Tensor:
        z_norm = torch.stack(
            [
                self.normalizer(flight_features[..., idx], col)
                for idx, col in enumerate(self.feature_cols)
            ],
            dim=-1,
        )
        # Each W1 weight = sign[i] · softplus(W1_raw[h, i]) → guarantees the
        # output is monotone non-decreasing in (sign[i] · feature[i]).
        w1 = self.signs[None, :] * functional.softplus(self.W1_raw)
        h0 = functional.linear(z_norm, w1, self.b1)
        h0 = functional.softplus(h0)
        # Positive weights on hidden layers preserve monotonicity.
        w2 = functional.softplus(self.W2_raw)
        h1 = functional.linear(h0, w2, self.b2)
        h1 = functional.softplus(h1)
        w3 = functional.softplus(self.W3_raw)
        z = (h1 * w3).sum(dim=-1) + self.b0
        alpha = torch.sigmoid(z / self.temperature)
        oew_kg = self.oew_kg.double()
        mtow_kg = self.mtow_kg.double()
        return oew_kg + alpha.double() * (mtow_kg - oew_kg)

    def effective_coefficients(self) -> dict[str, float]:
        """Return mean absolute input-layer weight per feature for diagnostics.

        Reports the average effective contribution of each input feature
        across hidden units, which acts as a non-linear analogue to the
        single coefficient :class:`MassEncoderLinear` exposes.
        """
        with torch.no_grad():
            w1 = self.signs[None, :] * functional.softplus(self.W1_raw)
            per_feature = w1.mean(dim=0)
            coefficients = {"b0": float(self.b0.item())}
            coefficients.update(
                {
                    col: float(value.item())
                    for col, value in zip(self.feature_cols, per_feature, strict=True)
                }
            )
        return coefficients
