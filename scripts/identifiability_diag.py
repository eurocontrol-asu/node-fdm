"""Shared diagnostics for the AXM-1740 absolute-score identifiability gate.

Reusable helpers consumed by :mod:`axm1740_recalibrate`,
:mod:`phase15_parity_report` and future Phase-2 / Phase-3 reports.

Provides:

* :func:`compute_sigma_obs` — empirical per-dx standard deviation over a
  validation dataset, used both as a diagnostic table in the markdown
  artifacts and as a sanity check against the trainer's internal
  ``sigma_obs_sq`` computation.
* :func:`format_identifiability_section` — stable markdown formatter for
  the absolute-score block embedded in every recalibration / parity
  report.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "compute_sigma_obs",
    "format_identifiability_section",
]


def compute_sigma_obs(val_dataset: Any, dx_cols: list[str]) -> dict[str, float]:
    """Compute the empirical standard deviation of each dx column over a val set.

    Args:
        val_dataset: A sequence-like dataset whose items expose a ``dx``
            tensor of shape ``(seq_len, n_dx)``. Typically a
            :class:`node_fdm.dataset.FlightDataset`.
        dx_cols: Ordered list of dx column names. Length must match
            ``dx.shape[-1]``.

    Returns:
        Dict mapping each dx col name to its empirical std (unbiased=False).

    Raises:
        ValueError: When ``dx_cols`` is empty (no columns to measure).
    """
    if not dx_cols:
        msg = "compute_sigma_obs requires a non-empty dx_cols list"
        raise ValueError(msg)

    dx_stack = torch.stack([val_dataset[i].dx for i in range(len(val_dataset))])
    dx_flat = dx_stack.reshape(-1, dx_stack.shape[-1])
    if dx_flat.shape[-1] != len(dx_cols):
        msg = f"dx tensor has {dx_flat.shape[-1]} columns but dx_cols lists {len(dx_cols)} names"
        raise ValueError(msg)

    stds = dx_flat.std(dim=0, unbiased=False)
    return {col: float(stds[i].item()) for i, col in enumerate(dx_cols)}


def format_identifiability_section(
    result: dict[str, float | bool],
    model_name: str,
    k_min: float | None = None,
) -> str:
    """Render the absolute-score block as a stable markdown section.

    Args:
        result: Dict returned by
            :meth:`node_fdm.trainer.ODETrainer.identifiability_test_absolute`.
            Must carry ``baseline_mse``, ``perturbed_mse``, ``delta_abs``,
            ``sigma_obs_sq``, ``absolute_score`` and ``passed_absolute``.
        model_name: Human-readable model identifier (e.g. ``full_hybrid_v2``).
        k_min: Optional acceptance floor. When ``None`` the verdict line
            reports ``n/a (calibration mode)`` and the ``k_min`` row is
            omitted from the table.

    Returns:
        Multi-line markdown string with a level-3 heading and a small
        key/value table.
    """
    baseline_mse = float(result["baseline_mse"])
    perturbed_mse = float(result["perturbed_mse"])
    delta_abs = float(result["delta_abs"])
    sigma_obs_sq = float(result["sigma_obs_sq"])
    absolute_score = float(result["absolute_score"])

    if k_min is None:
        verdict = "n/a (calibration mode)"
    else:
        verdict = "PASS" if absolute_score >= k_min else "FAIL"

    lines = [
        f"### Identifiability (absolute score) — {model_name}",
        "",
        "| Metric | Value |",
        "| -- | -- |",
        f"| baseline_mse | {baseline_mse:.6f} |",
        f"| perturbed_mse | {perturbed_mse:.6f} |",
        f"| delta_abs | {delta_abs:.6f} |",
        f"| sigma_obs_sq | {sigma_obs_sq:.6f} |",
        f"| absolute_score | {absolute_score:.6f} |",
    ]
    if k_min is not None:
        lines.append(f"| k_min | {k_min:.6f} |")
    lines.append(f"| verdict | {verdict} |")
    lines.append("")
    return "\n".join(lines)
