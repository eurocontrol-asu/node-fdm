"""Monotone linear mass encoder layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from node_fdm.layers.normalizers import InputNormalizer

__all__ = [
    "MassEncoderLinear",
    "MassEncoderLinearTempered",
    "MassEncoderMLPMonotone",
    "MassEncoderPSResidual",
]


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


class MassEncoderPSResidual(MassEncoderLinearTempered):
    """v9-style fallback + residual head anchored on Poll-Schumann Eq 100.

    For cruise-stable segments (alt >= 9000 m, |dh/dt| <= 1 m/s, FL in
    Eq 100 invertible band [342.7, 493.3]), the prediction is anchored
    on ``m_PS_anchor = mass_ratio(FL_obs) * MTOM``, computed at runtime
    from the pre-built Poll-Schumann optimum lookup table.

    For non-cruise segments, the anchor falls back to the v9-style
    ``MassEncoderLinearTempered`` (T=1.5, signs +/-/-).

    On top of the anchor, a small MLP head learns a residual
    ``delta_m_NN`` bounded to +/- ``delta_cap_kg`` via tanh, based on
    state at t0 ``(alt_norm, dh_dt_norm)``. The residual absorbs the
    ~14.5 % Eq 100 bias as a learned offset and provides local
    adaptation around the anchor.

    R5 compliant : ``alt_t0`` and ``dh_dt`` are derived from
    ``raw_alt_m`` already in the trainer's state tensor — no QAR data
    is consumed.

    Builds the Eq 100 inversion table at init using ``ps_core.optimum``
    (pure Python, no ML dependency) ; stores it as non-trainable
    buffers ``ps_fl_grid`` / ``ps_mr_grid``. The presence of the
    ``ps_fl_grid`` buffer is the duck-typing flag the trainer uses to
    detect this encoder class and pass the extra state arguments.
    """

    ps_fl_grid: torch.Tensor
    ps_mr_grid: torch.Tensor
    delta_cap_kg_buf: torch.Tensor

    def __init__(
        self,
        feature_stats: dict[str, dict[str, float]],
        feature_cols: list[str],
        expected_signs: list[float],
        oew_kg: float,
        mtow_kg: float,
        b0_init: float = 0.85,
        temperature: float = 1.5,
        delta_hidden: int = 16,
        delta_cap_kg: float = 5_000.0,
    ) -> None:
        super().__init__(
            feature_stats=feature_stats,
            feature_cols=feature_cols,
            expected_signs=expected_signs,
            oew_kg=oew_kg,
            mtow_kg=mtow_kg,
            b0_init=b0_init,
            temperature=temperature,
        )
        if delta_cap_kg <= 0.0:
            msg = f"delta_cap_kg must be > 0, got {delta_cap_kg}"
            raise ValueError(msg)
        self.delta_mlp = nn.Sequential(
            nn.Linear(2, delta_hidden),
            nn.SiLU(),
            nn.Linear(delta_hidden, 1),
            nn.Tanh(),
        )
        self.register_buffer(
            "delta_cap_kg_buf",
            torch.tensor(float(delta_cap_kg), dtype=torch.float32),
        )
        # Build the Poll-Schumann Eq 100 inversion table (A320 psi-set).
        # Pure-python ps_core lives outside the node-fdm-v2 repo ; import
        # via explicit sys.path injection.
        import sys as _sys

        ps_core_src = (
            "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages/ps-core/src"
        )
        if ps_core_src not in _sys.path:
            _sys.path.insert(0, ps_core_src)
        import numpy as _np
        from ps_core._types import AircraftPsi  # type: ignore[import-not-found]
        from ps_core.optimum import optimum_in_isa  # type: ignore[import-not-found]

        a320_psi = AircraftPsi(
            psi_1=0.156,
            psi_2=8.05,
            psi_4=0.753,
            psi_5=6.29e7,
            psi_6=0.656,
            tau=0.162,
        )
        mr = _np.linspace(0.5, 1.0, 1001, dtype=_np.float64)
        fl = _np.array(
            [optimum_in_isa(a320_psi, float(m)).fl_o for m in mr],
            dtype=_np.float64,
        )
        order = _np.argsort(fl)
        self.register_buffer(
            "ps_fl_grid",
            torch.tensor(fl[order], dtype=torch.float32),
        )
        self.register_buffer(
            "ps_mr_grid",
            torch.tensor(mr[order], dtype=torch.float32),
        )

    def _ps_anchor(
        self, alt_t0: torch.Tensor, dh_dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (m_PS_anchor_kg, cruise_mask). Both shape (batch,)."""
        fl_grid = self.ps_fl_grid
        mr_grid = self.ps_mr_grid
        fl_min, fl_max = fl_grid[0], fl_grid[-1]
        fl_obs = alt_t0 * (1.0 / 0.3048) / 100.0
        fl_clamped = fl_obs.clamp(min=fl_min.item(), max=fl_max.item())
        idx = torch.searchsorted(fl_grid, fl_clamped).clamp(min=1, max=len(fl_grid) - 1)
        fl_lo = fl_grid[idx - 1]
        fl_hi = fl_grid[idx]
        mr_lo = mr_grid[idx - 1]
        mr_hi = mr_grid[idx]
        t = (fl_clamped - fl_lo) / (fl_hi - fl_lo).clamp(min=1e-9)
        mass_ratio = mr_lo + t * (mr_hi - mr_lo)
        m_ps = mass_ratio * self.mtow_kg.float()
        in_range = (fl_obs >= fl_min) & (fl_obs <= fl_max)
        is_cruise = (alt_t0 >= 9000.0) & (dh_dt.abs() <= 1.0) & in_range
        return m_ps, is_cruise

    def forward(  # type: ignore[override]
        self,
        flight_features: torch.Tensor,
        alt_t0: torch.Tensor | None = None,
        dh_dt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode (flight_features, alt_t0, dh_dt) into m_predicted.

        When ``alt_t0`` / ``dh_dt`` are None (e.g. legacy callers), falls
        back to the v9 linear-tempered encoder for backward compatibility.
        """
        m_fallback = super().forward(flight_features)
        if alt_t0 is None or dh_dt is None:
            return m_fallback
        m_ps_anchor, is_cruise = self._ps_anchor(alt_t0, dh_dt)
        m_fallback_f = m_fallback.to(m_ps_anchor.dtype)
        anchor = torch.where(is_cruise, m_ps_anchor, m_fallback_f)
        # Δm head : MLP on normalised state features.
        alt_norm = alt_t0 / 12_000.0  # ~FL400 scale
        dh_dt_norm = dh_dt / 10.0  # |dh/dt| up to ~10 m/s in climb
        state_feats = torch.stack([alt_norm, dh_dt_norm], dim=-1).to(
            torch.float32,
        )
        delta_unit = self.delta_mlp(state_feats).squeeze(-1)
        delta_kg = delta_unit.to(anchor.dtype) * self.delta_cap_kg_buf.to(
            anchor.dtype,
        )
        m_pred = (anchor + delta_kg).clamp(
            min=self.oew_kg.float(),
            max=self.mtow_kg.float(),
        )
        return m_pred.to(m_fallback.dtype)

    def effective_coefficients(self) -> dict[str, float]:
        """Report the fallback's coefficients + residual head magnitude."""
        coefs = super().effective_coefficients()
        with torch.no_grad():
            # Approx residual magnitude : sample 256 random (alt, dh_dt) points
            # in the cruise envelope and look at the absolute Δm output.
            n = 256
            alt = torch.linspace(9_500.0, 12_500.0, n)
            dh_dt = torch.zeros(n)
            alt_norm = alt / 12_000.0
            dh_dt_norm = dh_dt / 10.0
            feats = torch.stack([alt_norm, dh_dt_norm], dim=-1)
            delta_unit = self.delta_mlp(feats).squeeze(-1)
            delta_kg = delta_unit * self.delta_cap_kg_buf.to(delta_unit.dtype)
        coefs["delta_cruise_mean_kg"] = float(delta_kg.mean().item())
        coefs["delta_cruise_absmean_kg"] = float(delta_kg.abs().mean().item())
        return coefs


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
