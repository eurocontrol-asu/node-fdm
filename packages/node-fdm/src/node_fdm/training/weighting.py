"""Effective-number sample weights for class-imbalanced training.

Flight-mode labels (~13 modes: TURN, ALT_MACH, ALT_CAS, VS_CAS, ...) are
heavily skewed toward cruise (`ALT_MACH`) — typical run shows 100x to 1000x
ratio between dominant and rare modes. Without rebalancing, the rollout
loss is dominated by cruise samples and the model under-fits transitions
(turns, climbs, descents).

We use the **Effective Number of Samples** weighting from Cui et al. 2019
("Class-Balanced Loss Based on Effective Number of Samples", CVPR'19):

    N_eff[L] = (1 - beta**count[L]) / (1 - beta)
    w_raw[L] = 1 / N_eff[L]

The per-sample weight is then normalised so that the dataset-weighted mean
equals 1, i.e. summing `count[L] * w[L]` over all labels recovers the total
sample count. This keeps the average loss magnitude unchanged — only the
relative contribution of each mode shifts.

beta is selected by `auto_beta`: `beta = clip(1 - 1/sqrt(ratio), 0.99, 0.9999)`
where `ratio = max_count / min_count`. The intuition is that beta -> 1 as
the imbalance ratio grows, calibrating the strength of the rebalance to the
actual skew. `cui_beta` exposes the alternative `1 - 10/N_total` heuristic
from the original paper for cross-validation logging.

Numerical note: `(1 - beta**count) / (1 - beta)` is computed in float64.
For beta ~ 0.9999 and count ~ 1e7, `beta**count -> 0` and the ratio caps
at ~ 1/(1-beta) ~ 1e4, well within float64 dynamic range. Per-sample
weights are cast to Float32 only at attach time.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
import structlog

__all__ = [
    "attach_sample_weights",
    "auto_beta",
    "boot_mode_weights",
    "compute_mode_weights",
    "cui_beta",
]

log = structlog.get_logger(__name__)

_BETA_LO = 0.99
_BETA_HI = 0.9999
_LABEL_COL = "fdm_mode_label"
_WEIGHT_COL = "fdm_train_weight"


def auto_beta(counts: dict[str, int]) -> float:
    """Pick a beta calibrated to the observed imbalance ratio.

    # why: beta = 1 - 1/sqrt(ratio) -> beta->1 as ratio->inf
    # (calibrate strength of rebalance to the actual skew)

    >>> auto_beta({"A": 100, "B": 90})
    0.99
    >>> round(auto_beta({"A": 1_000_000, "B": 1_000}), 6)
    0.968377
    """
    if len(counts) < 2:
        msg = f"auto_beta requires at least 2 labels, got {len(counts)}"
        raise ValueError(msg)
    values = list(counts.values())
    if min(values) <= 0:
        msg = f"auto_beta requires strictly positive counts, got min={min(values)}"
        raise ValueError(msg)
    ratio = max(values) / min(values)
    beta = 1.0 - 1.0 / math.sqrt(ratio)
    return max(_BETA_LO, min(_BETA_HI, beta))


def cui_beta(n_total: int) -> float:
    """Cui 2019 default heuristic: ``1 - 10/N_total``. Informational only."""
    if n_total <= 10:
        msg = f"cui_beta requires n_total > 10, got {n_total}"
        raise ValueError(msg)
    return 1.0 - 10.0 / n_total


def compute_mode_weights(counts: dict[str, int]) -> dict[str, float]:
    """Return per-label weights from Cui 2019 effective-number formula.

    For each label L with count[L] >= 1:
        N_eff[L] = (1 - beta**count[L]) / (1 - beta)
        w_raw[L] = 1 / N_eff[L]

    Weights are then normalised so the dataset-weighted mean is 1:
        sum(count[L] * w[L]) == sum(count[L])

    Labels with ``count == 0`` are **omitted** from the output (no NaN, no
    inf, no division by zero). Determinism: input dict insertion order is
    preserved in the output.

    >>> w = compute_mode_weights({"rare": 100, "common": 100_000})
    >>> w["rare"] > w["common"]
    True
    """
    nonzero = {k: int(v) for k, v in counts.items() if v > 0}
    if not nonzero:
        return {}
    beta = auto_beta(nonzero) if len(nonzero) >= 2 else _BETA_LO
    one_minus_beta = 1.0 - beta
    raw: dict[str, float] = {}
    for label, c in nonzero.items():
        n_eff = (1.0 - beta**c) / one_minus_beta
        raw[label] = 1.0 / n_eff
    total = sum(nonzero.values())
    weighted_sum = sum(nonzero[k] * raw[k] for k in nonzero)
    norm = total / weighted_sum
    return {k: raw[k] * norm for k in nonzero}


def attach_sample_weights(df: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    """Add a ``fdm_train_weight: Float32`` column by mapping ``fdm_mode_label``.

    Raises:
        KeyError: If a row's label is missing from *weights*. The message
            contains the offending label.
    """
    labels = df.get_column(_LABEL_COL).to_list()
    out: list[float] = []
    for label in labels:
        if label not in weights:
            msg = f"{label!r} missing from weights dict"
            raise KeyError(msg)
        out.append(weights[label])
    return df.with_columns(pl.Series(_WEIGHT_COL, out, dtype=pl.Float32))


def _top_k(
    weights: dict[str, float], counts: dict[str, int], k: int, *, descending: bool
) -> list[list[Any]]:
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=descending)
    return [[label, float(w), int(counts[label])] for label, w in items[:k]]


def boot_mode_weights(train_df: pl.DataFrame) -> pl.DataFrame:
    """Compute mode weights and attach them to *train_df*; emit boot log.

    Counts labels in `train_df["fdm_mode_label"]`, computes beta + weights,
    appends `fdm_train_weight`, and logs one ``mode_weights_computed`` event
    carrying the diagnostic fields required by the spec.
    """
    counts_series = train_df.get_column(_LABEL_COL).value_counts()
    counts: dict[str, int] = {
        row[0]: int(row[1]) for row in counts_series.iter_rows() if row[0] is not None
    }
    weights = compute_mode_weights(counts)
    out = attach_sample_weights(train_df, weights)
    nonzero_counts = {k: v for k, v in counts.items() if v > 0}
    n_total = sum(nonzero_counts.values())
    nonzero_vals = list(nonzero_counts.values())
    imbalance = max(nonzero_vals) / min(nonzero_vals) if len(nonzero_vals) >= 2 else 1.0
    weight_vals = list(weights.values())
    log.info(
        "mode_weights_computed",
        beta=auto_beta(nonzero_counts) if len(nonzero_counts) >= 2 else _BETA_LO,
        beta_cui=cui_beta(n_total) if n_total > 10 else None,
        imbalance_ratio=imbalance,
        n_labels=len(weights),
        top5_labels=_top_k(weights, counts, 5, descending=True),
        bottom5_labels=_top_k(weights, counts, 5, descending=False),
        weight_ratio_max_over_min=(max(weight_vals) / min(weight_vals) if weight_vals else 1.0),
    )
    return out
