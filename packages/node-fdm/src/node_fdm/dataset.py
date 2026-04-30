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

from node_fdm_data.physics.constants import G, R
from node_fdm_data.physics.isa import isa_pressure, isa_temperature

__all__ = [
    "DERIVED_FEATURES",
    "FlightDataset",
    "FlightSample",
    "compute_stats",
]

# ---------------------------------------------------------------------------
# Derived feature computers (mirrors TrajectoryLayer.forward formulas exactly)
# ---------------------------------------------------------------------------
# Each function signature: (x_arr, e_arr, x_cols, e_cols) -> np.ndarray
# x_arr shape: (N, n_x), e_arr shape: (N, n_e), column lists give semantics.
# ---------------------------------------------------------------------------


def _compute_g_sin_gamma(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
) -> np.ndarray:
    """G * sin(gamma) — mirrors ``output[c["g_sin_gamma"]]`` in TrajectoryLayer."""
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")]
    return np.asarray(G * np.sin(gamma), dtype=np.float64)


def _compute_cos_gamma(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
) -> np.ndarray:
    """cos(gamma) — mirrors ``output[c["cos_gamma"]]`` in TrajectoryLayer."""
    gamma = x_arr[:, x_cols.index("fdm_gamma_rad")]
    return np.asarray(np.cos(gamma), dtype=np.float64)


def _compute_g_over_v(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
) -> np.ndarray:
    """G / max(tas, 1.0) — mirrors ``output[c["g_over_v"]]`` in TrajectoryLayer."""
    tas = x_arr[:, x_cols.index("era_tas_ms")]
    return np.asarray(G / np.maximum(tas, 1.0), dtype=np.float64)


def _compute_q(
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    x_cols: list[str],
    e_cols: list[str],
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


#: Registry of derived e1 feature computers keyed by column name.
#: Each callable has signature
#: ``(x_arr, e_arr, x_cols, e_cols) -> np.ndarray``.
_DerivedFn = Callable[
    [np.ndarray, np.ndarray, list[str], list[str]],
    np.ndarray,
]
DERIVED_FEATURES: dict[str, _DerivedFn] = {
    "fdm_g_sin_gamma_ms2": _compute_g_sin_gamma,
    "fdm_cos_gamma": _compute_cos_gamma,
    "fdm_g_over_v": _compute_g_over_v,
    "fdm_q_pa": _compute_q,
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
    """

    x: torch.Tensor
    u: torch.Tensor
    e: torch.Tensor
    dx: torch.Tensor
    e1: torch.Tensor | None = field(default=None)


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
) -> dict[str, dict[str, float]]:
    """Compute per-column statistics from a list of samples.

    Returns a mapping ``column_name → {"mean": ..., "std": ..., "max": ...}``
    that can be passed directly to ``FlightDynamicsModel`` / ``ModelMeta``.

    Args:
        samples: List of flight samples to aggregate.
        x_cols: State column names.
        u_cols: Control column names.
        e_cols: Environment column names.
        dx_cols: Derivative column names.
        e1_cols: Optional extra environment column names.

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
                stats[col] = {
                    "mean": clean.mean().item() if len(clean) > 0 else 0.0,
                    "std": (clean.std().item() if len(clean) > 1 else 0.0) + 1e-6,
                    "max": clean.abs().max().item() if len(clean) > 0 else 0.0,
                }

        # Compute derived features analytically for columns not yet in stats
        for col in e1_cols:
            if col in stats:
                continue
            if col not in DERIVED_FEATURES:
                continue
            compute_fn = DERIVED_FEATURES[col]
            arrays: list[np.ndarray] = []
            for s in samples:
                x_np = s.x.numpy().astype(np.float64)
                e_np = s.e.numpy().astype(np.float64)
                vals_np = compute_fn(x_np, e_np, x_cols, e_cols)
                arrays.append(vals_np.ravel())
            all_vals = np.concatenate(arrays)
            finite = np.isfinite(all_vals)
            clean_np = all_vals[finite] if not finite.all() else all_vals
            if len(clean_np) > 0:
                stats[col] = {
                    "mean": float(clean_np.mean()),
                    "std": float(clean_np.std()) + 1e-6,
                    "max": float(np.abs(clean_np).max()),
                }

    return stats
