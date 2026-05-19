"""Flight dataset with typed samples for Neural ODE training.

Replaces the legacy ``SeqDataset(Dataset[dict[str, Tensor]])`` with a
properly typed ``FlightDataset(Dataset[FlightSample])`` that returns
frozen dataclass instances.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from node_fdm.layers.physics import S_REF_A320_M2, cl_ref_steady_np
from node_fdm_data.physics.constants import G, R
from node_fdm_data.physics.isa import isa_pressure, isa_temperature
from node_fdm_data.schemas.adsb_hybrid import A320_MTOW_KG, A320_OEW_KG

__all__ = [
    "DERIVED_FEATURES",
    "FlightDataset",
    "FlightSample",
    "compute_stats",
]

# ---------------------------------------------------------------------------
# Derived feature computers (mirror TrajectoryLayer / inverse-PhysicsLayer formulas)
# ---------------------------------------------------------------------------
# Each function signature:
#   (x_arr, e_arr, dx_arr, x_cols, e_cols, dx_cols) -> np.ndarray
# x_arr  shape: (N, n_x), e_arr shape: (N, n_e), dx_arr shape: (N, n_dx).
# Column lists give semantics; functions that don't need dx ignore it.
# ---------------------------------------------------------------------------


# Lower bound on TAS for the ``1/V`` term in NN-output inversion.
# Must match ``layers.physics.V_MIN_CLAMP`` so derived stats reflect the
# exact algebraic inverse of what the PhysicsLayer applies at runtime.
_V_MIN_CLAMP: float = 50.0
_M_REF_KG: float = (A320_OEW_KG + A320_MTOW_KG) / 2.0


def _compute_g_sin_gamma(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """G * sin(gamma) — mirrors ``output[c["g_sin_gamma"]]`` in TrajectoryLayer."""
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")]
    return np.asarray(G * np.sin(gamma), dtype=np.float64)


def _compute_cos_gamma(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """cos(gamma) — mirrors ``output[c["cos_gamma"]]`` in TrajectoryLayer."""
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")]
    return np.asarray(np.cos(gamma), dtype=np.float64)


def _compute_g_over_v(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """G / max(tas, 1.0) — mirrors ``output[c["g_over_v"]]`` in TrajectoryLayer."""
    tas = x_arr[:, x_cols.index("era_tas_ms")]
    return np.asarray(G / np.maximum(tas, 1.0), dtype=np.float64)


def _compute_q(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """0.5 * rho * V^2 — mirrors ``output[c["q"]]`` in TrajectoryLayer.

    Uses ERA5 temperature when available (same as TrajectoryLayer), else ISA.
    """
    alt = x_arr[:, x_cols.index("raw_alt_m")]
    tas = x_arr[:, x_cols.index("era_tas_ms")]
    if "era_temp_K" in e_cols:
        temp = e_arr[:, e_cols.index("era_temp_K")].astype(np.float64)
        temp = np.nan_to_num(temp, nan=288.15, posinf=320.0, neginf=150.0)
        temp = np.clip(temp, 150.0, 320.0)
    else:
        temp = np.asarray(isa_temperature(alt.astype(np.float64)), dtype=np.float64)
    p = np.asarray(isa_pressure(alt.astype(np.float64)), dtype=np.float64)
    rho = p / (R * temp)
    return np.asarray(0.5 * rho * tas.astype(np.float64) ** 2, dtype=np.float64)


# --- Inverse PhysicsLayer (NN-output targets) ------------------------------
# These columns are produced by the trainable StructuredLayer at runtime and
# are absent from the dataset. We compute them analytically by inverting the
# PhysicsLayer equations so compute_stats can derive their (mean, std, p999)
# directly from observable derivatives.
#
#   a_spec       = d_tas + g*sin(gamma)
#   n_z_residual = (V_safe/g)*d_gamma + cos(gamma) - 1   V_safe = max(tas, 50)
#   phi_bank     = atan( (V_safe/g)*d_heading )


def _compute_a_spec(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """Inverse PhysicsLayer for ``fdm_a_spec_ms2 = d_tas + g*sin(gamma)``."""
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")].astype(np.float64)
    d_tas = dx_arr[:, dx_cols.index("fdm_d_tas_ms2")].astype(np.float64)
    return np.asarray(d_tas + G * np.sin(gamma), dtype=np.float64)


def _compute_n_z_residual(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """Inverse PhysicsLayer for ``fdm_n_z_residual = (V/g)*d_gamma + cos(gamma) - 1``."""
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")].astype(np.float64)
    tas = x_arr[:, x_cols.index("era_tas_ms")].astype(np.float64)
    d_gamma = dx_arr[:, dx_cols.index("fdm_d_gamma_rads")].astype(np.float64)
    v_safe = np.maximum(tas, _V_MIN_CLAMP)
    return np.asarray((v_safe / G) * d_gamma + np.cos(gamma) - 1.0, dtype=np.float64)


def _compute_t_minus_d_norm(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """Inverse PhysicsLayer for the hybrid adim output.

    PhysicsLayer reconstructs ``T-D = t_minus_d_norm * m_ref`` then applies
    ``d_TAS = (T-D)/m - g*sin(gamma)``. Solving for the NN target under the
    convention that the MassEncoder absorbs the per-flight residue via the
    division by ``m`` at runtime:

        t_minus_d_norm = (d_TAS + g*sin(gamma)) * m / m_ref

    For statistics we use ``m = m_ref`` (the dataset has no observed mass)
    so the computer collapses to ``d_TAS + g*sin(gamma)`` — i.e. the same
    target as the legacy ``a_spec``. The scale comes out comparable to the
    baseline p99.9 ≈ 1 m/s².
    """
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")].astype(np.float64)
    d_tas = dx_arr[:, dx_cols.index("fdm_d_tas_ms2")].astype(np.float64)
    return np.asarray(d_tas + G * np.sin(gamma), dtype=np.float64)


def _compute_lift_residual_norm(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """Inverse PhysicsLayer for the hybrid adim lift residual.

    PhysicsLayer reconstructs ``L = lift_residual_norm * m_ref*g + m*g``
    then applies ``d_gamma = (L/m - g*cos gamma) / V``. Solving and using
    ``m = m_ref`` for statistics yields the same target as the legacy
    ``n_z_residual = (V/g)*d_gamma + cos(gamma) - 1``.
    """
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")].astype(np.float64)
    tas = x_arr[:, x_cols.index("era_tas_ms")].astype(np.float64)
    d_gamma = dx_arr[:, dx_cols.index("fdm_d_gamma_rads")].astype(np.float64)
    v_safe = np.maximum(tas, _V_MIN_CLAMP)
    return np.asarray((v_safe / G) * d_gamma + np.cos(gamma) - 1.0, dtype=np.float64)


def _compute_cl_residual(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """Inverse PhysicsLayer for the CL-mode lift residual.

    PhysicsLayer reconstructs ``L = q · S · (CL_steady(q) + cl_residual)``
    where ``CL_steady(q) = m_ref · g / (q · S)`` balances weight at the
    current dynamic pressure. Applying ``d_gamma = (L/m - g·cos gamma) / V``
    and solving for the NN target under the ``m = m_ref`` statistics
    convention:

        cl_residual = ((V_safe/G · d_gamma + cos gamma) · m_ref · G)
                      / (q · S_REF_A320_M2) - CL_steady(q)

    where ``q`` is the dynamic pressure already computed by ``_compute_q``
    (ERA5 temperature when available, ISA fallback). The q-dependent
    baseline (vs the former CL_REF=0.5 constant) keeps the target residual
    centered near 0 in every phase — see AXM-1739 cl_distribution.md §6.
    """
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")].astype(np.float64)
    tas = x_arr[:, x_cols.index("era_tas_ms")].astype(np.float64)
    d_gamma = dx_arr[:, dx_cols.index("fdm_d_gamma_rads")].astype(np.float64)
    v_safe = np.maximum(tas, _V_MIN_CLAMP)
    q_pa = _compute_q(x_arr, e_arr, dx_arr, x_cols, e_cols, dx_cols)
    lhs = ((v_safe / G) * d_gamma + np.cos(gamma)) * _M_REF_KG * G
    cl_ref_q = cl_ref_steady_np(q_pa)
    return np.asarray(lhs / (q_pa * S_REF_A320_M2) - cl_ref_q, dtype=np.float64)


def _compute_phi_bank(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    dx_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
) -> np.ndarray:
    """Inverse PhysicsLayer for ``fdm_phi_bank_rad = atan((V/g)·d_heading)``."""
    tas = x_arr[:, x_cols.index("era_tas_ms")].astype(np.float64)
    d_heading = dx_arr[:, dx_cols.index("fdm_d_heading_rads")].astype(np.float64)
    v_safe = np.maximum(tas, _V_MIN_CLAMP)
    return np.asarray(np.arctan((v_safe / G) * d_heading), dtype=np.float64)


#: Registry of derived feature computers keyed by column name.
#: Each callable has signature
#: ``(x_arr, e_arr, dx_arr, x_cols, e_cols, dx_cols) -> np.ndarray``.
_DerivedFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]],
    np.ndarray,
]
DERIVED_FEATURES: dict[str, _DerivedFn] = {
    # TrajectoryLayer mirror (kinematic e1 features).
    "fdm_g_sin_gamma_ms2": _compute_g_sin_gamma,
    "fdm_cos_gamma": _compute_cos_gamma,
    "fdm_g_over_v": _compute_g_over_v,
    "fdm_q_pa": _compute_q,
    # Inverse PhysicsLayer (NN-output targets — used to derive p999 caps).
    "fdm_a_spec_ms2": _compute_a_spec,
    "fdm_n_z_residual": _compute_n_z_residual,
    # Hybrid arch: adim outputs reconstructed with mass in PhysicsLayer.
    "fdm_t_minus_d_norm": _compute_t_minus_d_norm,
    "fdm_lift_residual_norm": _compute_lift_residual_norm,
    "fdm_cl_residual": _compute_cl_residual,
    "fdm_phi_bank_rad": _compute_phi_bank,
}


@dataclass(frozen=True)
class FlightSample:
    """Single training sample containing windowed flight data tensors.

    Attributes:
        x: State tensor of shape ``(seq_len, n_x)``.
        u: Control tensor of shape ``(seq_len, n_u)``.
        e: Environment tensor of shape ``(seq_len, n_e)``.
        dx: Derivative tensor of shape ``(seq_len, n_dx)``.
        e1: Optional extra environment tensor of shape ``(seq_len, n_e1)``.
        w: Optional per-sample training weight of shape ``(seq_len,)``
            populated when mode-balanced loss weighting is enabled.
        flight_features: Optional flight-level feature tensor of shape
            ``(seq_len, n_features)``.
    """

    x: torch.Tensor
    u: torch.Tensor
    e: torch.Tensor
    dx: torch.Tensor
    e1: torch.Tensor | None = field(default=None)
    w: torch.Tensor | None = field(default=None)
    flight_features: torch.Tensor | None = field(default=None)


class FlightDataset(Dataset[FlightSample]):
    """Dataset of pre-built flight samples.

    Accepts an already-constructed list of :class:`FlightSample` instances.
    File I/O and windowing are handled upstream (e.g. in ``loader.py``).

    Args:
        samples: Non-empty list of flight samples.

    Raises:
        ValueError: If *samples* is empty.
    """

    def __init__(self, samples: list[FlightSample]) -> None:
        if not samples:
            msg = "FlightDataset requires at least one sample, got 0."
            raise ValueError(msg)
        self._samples = samples

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> FlightSample:
        """Return the sample at *idx*."""
        return self._samples[idx]


def compute_stats(
    samples: list[FlightSample],
    x_cols: list[str],
    u_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
    *,
    e1_cols: list[str] | None = None,
    flight_feature_cols: list[str] | None = None,
    derived_cols: list[str] | None = None,
    derived_scale_floor_ratio: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Compute per-column statistics from a list of samples.

    Returns a mapping ``column_name → {"mean", "std", "max", "p999", "iqr"}``
    that can be passed directly to ``FlightDynamicsModel`` / ``ModelMeta``.

    Args:
        samples: List of flight samples to aggregate.
        x_cols: State column names.
        u_cols: Control column names.
        e_cols: Environment column names.
        dx_cols: Derivative column names.
        e1_cols: Optional extra environment column names. When provided, each
            column is sourced from ``s.e1`` (positional) when available, else
            falls back to a ``DERIVED_FEATURES`` analytic computer.
        flight_feature_cols: Optional flight-level feature column names. When
            provided, each column is sourced positionally from
            ``s.flight_features`` when available.
        derived_cols: Optional list of NN-output / derived columns to compute
            purely from ``DERIVED_FEATURES``. Used for stats on quantities
            that the trainable layer emits (e.g. ``fdm_a_spec_ms2``) but that
            never appear in the dataset; their p999 feeds the
            ``OutputDenormalizer`` scale via ``_create_structured_layer``.
        derived_scale_floor_ratio: When > 0, the p999 of each derived column
            is computed on the *conditional* tail ``|x| > ratio * p999_uncond``
            instead of the full distribution. Removes dilution from
            near-zero samples (cruise / straight flight) so the resulting
            scale reflects the natural unit of the active signal.

    Returns:
        Per-column statistics dictionary.
    """
    all_cols = x_cols + u_cols + e_cols + dx_cols

    # Concatenate all samples into one big tensor per category
    x_all = torch.cat([s.x for s in samples], dim=0)
    e_all = torch.cat([s.e for s in samples], dim=0)
    dx_all = torch.cat([s.dx for s in samples], dim=0)
    parts = [x_all]
    if u_cols:
        u_all = torch.cat([s.u for s in samples], dim=0)
        parts.append(u_all)
    parts.extend([e_all, dx_all])
    data = torch.cat(parts, dim=1)

    stats: dict[str, dict[str, float]] = {}
    for i, col in enumerate(all_cols):
        vals = data[:, i]
        p05 = torch.quantile(vals, 0.005).item()
        p995 = torch.quantile(vals, 0.995).item()
        stats[col] = {
            "mean": vals.mean().item(),
            "std": vals.std().item() + 1e-6,
            "max": vals.abs().max().item(),
            "p999": torch.quantile(vals.abs(), 0.999).item(),
            "iqr": max(p995 - p05, 1e-6),
        }

    # Append extra E1 columns if provided
    if e1_cols:
        e1_tensors = [s.e1 for s in samples if s.e1 is not None]
        if e1_tensors:
            e1_all = torch.cat(e1_tensors, dim=0)
            # Only iterate over columns actually present in the tensor
            # (loader may skip missing columns)
            n_e1 = e1_all.shape[1]
            for i, col in enumerate(e1_cols[:n_e1]):
                if col in stats:
                    continue  # DX/E stats take precedence over E1
                vals = e1_all[:, i]
                finite_mask = vals.isfinite()
                clean = vals[finite_mask] if not finite_mask.all() else vals
                if len(clean) > 0:
                    abs_clean = clean.abs()
                    stats[col] = {
                        "mean": clean.mean().item(),
                        "std": (clean.std().item() if len(clean) > 1 else 0.0) + 1e-6,
                        "max": abs_clean.max().item(),
                        "p999": torch.quantile(abs_clean, 0.999).item(),
                    }
                else:
                    stats[col] = {"mean": 0.0, "std": 1e-6, "max": 0.0, "p999": 0.0}

        # Compute derived e1 features analytically for columns not yet in stats
        for col in e1_cols:
            if col in stats:
                continue
            if col not in DERIVED_FEATURES:
                continue
            stats[col] = _compute_derived_stats(samples, col, x_cols, e_cols, dx_cols)

    if flight_feature_cols:
        flight_feature_tensors = [
            s.flight_features for s in samples if s.flight_features is not None
        ]
        if flight_feature_tensors:
            flight_features_all = torch.cat(flight_feature_tensors, dim=0)
            n_flight_features = flight_features_all.shape[1]
            for i, col in enumerate(flight_feature_cols[:n_flight_features]):
                vals = flight_features_all[:, i]
                abs_vals = vals.abs()
                stats[col] = {
                    "mean": vals.mean().item(),
                    "std": vals.std().item() + 1e-6,
                    "max": abs_vals.max().item(),
                    "p999": torch.quantile(abs_vals, 0.999).item(),
                }
        else:
            for col in flight_feature_cols:
                stats[col] = {"mean": 0.0, "std": 1e-6, "max": 0.0, "p999": 0.0}

    # Pure NN-output derived columns (never present in any tensor; fed to
    # _create_structured_layer for OutputDenormalizer scale via p999).
    if derived_cols:
        for col in derived_cols:
            if col in stats:
                continue
            if col not in DERIVED_FEATURES:
                msg = f"derived column '{col}' has no entry in DERIVED_FEATURES"
                raise KeyError(msg)
            stats[col] = _compute_derived_stats(
                samples,
                col,
                x_cols,
                e_cols,
                dx_cols,
                scale_floor_ratio=derived_scale_floor_ratio,
            )

    return stats


def _compute_derived_stats(
    samples: list[FlightSample],
    col: str,
    x_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
    *,
    scale_floor_ratio: float = 0.0,
) -> dict[str, float]:
    """Aggregate ``mean/std/max/p999`` for a derived column.

    Runs the ``DERIVED_FEATURES`` computer over every sample, concatenates,
    drops NaN/Inf rows, and returns the standard four-stat dict. Empty
    finite mask falls back to the neutral ``{0, 1e-6, 0, 0}`` triple — same
    contract as the e1-tensor branch.

    When ``scale_floor_ratio > 0``, the ``p999`` is computed on the
    *conditional* tail ``|x| > ratio * p999_unconditional`` to remove the
    dilution caused by near-zero samples (cruise, straight flight).
    """
    compute_fn = DERIVED_FEATURES[col]
    arrays: list[np.ndarray] = []
    for s in samples:
        x_np = s.x.numpy().astype(np.float64)
        e_np = s.e.numpy().astype(np.float64)
        dx_np = s.dx.numpy().astype(np.float64)
        vals_np = compute_fn(x_np, e_np, dx_np, x_cols, e_cols, dx_cols)
        arrays.append(vals_np.ravel())
    all_vals = np.concatenate(arrays)
    finite = np.isfinite(all_vals)
    clean_np = all_vals[finite] if not finite.all() else all_vals
    if len(clean_np) == 0:
        return {"mean": 0.0, "std": 1e-6, "max": 0.0, "p999": 0.0}
    abs_clean = np.abs(clean_np)
    p999_uncond = float(np.quantile(abs_clean, 0.999))
    if scale_floor_ratio > 0.0:
        threshold = scale_floor_ratio * p999_uncond
        active = abs_clean[abs_clean > threshold]
        # Need a minimum number of active samples to compute a stable p99.9;
        # fall back to unconditional when the column is essentially zero.
        p999 = float(np.quantile(active, 0.999)) if len(active) >= 1000 else p999_uncond
    else:
        p999 = p999_uncond
    return {
        "mean": float(clean_np.mean()),
        "std": float(clean_np.std()) + 1e-6,
        "max": float(abs_clean.max()),
        "p999": p999,
    }
