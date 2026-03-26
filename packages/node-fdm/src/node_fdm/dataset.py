"""Flight dataset with typed samples for Neural ODE training.

Replaces the legacy ``SeqDataset(Dataset[dict[str, Tensor]])`` with a
properly typed ``FlightDataset(Dataset[FlightSample])`` that returns
frozen dataclass instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

__all__ = [
    "FlightDataset",
    "FlightSample",
    "compute_stats",
]


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
        stats[col] = {
            "mean": vals.mean().item(),
            "std": vals.std().item() + 1e-6,
            "max": vals.abs().max().item(),
            "p999": torch.quantile(vals.abs(), 0.999).item(),
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
                vals = e1_all[:, i]
                finite_mask = vals.isfinite()
                clean = vals[finite_mask] if not finite_mask.all() else vals
                stats[col] = {
                    "mean": clean.mean().item() if len(clean) > 0 else 0.0,
                    "std": (clean.std().item() if len(clean) > 1 else 0.0) + 1e-6,
                    "max": clean.abs().max().item() if len(clean) > 0 else 0.0,
                }

    return stats
