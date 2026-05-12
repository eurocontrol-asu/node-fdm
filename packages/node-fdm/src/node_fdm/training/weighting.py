"""Power-law sample weights for class-imbalanced training.

Flight-mode labels (~13 modes: TURN, ALT_MACH, ALT_CAS, VS_CAS, ...) are
heavily skewed toward cruise (`ALT_MACH`) — typical run shows 40x to 100x
ratio between dominant and rare modes. Without rebalancing, the rollout
loss is dominated by cruise samples and the model under-fits transitions
(turns, climbs, descents).

We use a **power-law** weighting:

    w_raw[L] = 1 / count[L] ** alpha

where ``alpha`` controls the strength of the rebalance:

- ``alpha = 0``    -> uniform (no reweighting)
- ``alpha = 0.5``  -> sqrt inverse-frequency, standard NLP/detection
- ``alpha = 1``    -> full inverse-frequency, aggressive

The per-label weight is then normalised so that the dataset-weighted mean
equals 1, i.e. summing ``count[L] * w[L]`` over all labels recovers the
total sample count. This keeps the average loss magnitude unchanged —
only the relative contribution of each mode shifts.

History: an earlier implementation used the Cui 2019 effective-number
formula. For datasets with counts >> 10^3 per class (our case), the
required ``beta`` saturated at the clipping bound and all weights
collapsed to 1.0 — a silent no-op. Power-law is scale-invariant and has
no such failure mode.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import structlog
import torch

__all__ = [
    "attach_sample_weights",
    "boot_mode_weights",
    "compute_mode_weights",
    "compute_segment_weights",
]

log = structlog.get_logger(__name__)

_LABEL_COL = "fdm_mode_label"
_WEIGHT_COL = "fdm_train_weight"
_DEFAULT_ALPHA = 0.5


def compute_segment_weights(weights_per_sample: torch.Tensor) -> torch.Tensor:
    """Aggregate per-sample weights into per-segment weights via mean.

    The spec requires arithmetic mean over the time axis (not majority
    vote, not ``any``) so that 50/50 batches resolve to the average of
    both modes' weights.

    Args:
        weights_per_sample: Tensor of shape ``(batch, seq_len)``.

    Returns:
        Tensor of shape ``(batch,)`` with one weight per segment.
    """
    return weights_per_sample.mean(dim=-1)


def compute_mode_weights(
    counts: dict[str, int], alpha: float = _DEFAULT_ALPHA
) -> dict[str, float]:
    """Return per-label power-law weights, normalised to weighted-mean 1.

    For each label L with ``count[L] >= 1``:

        w_raw[L] = 1 / count[L] ** alpha

    Weights are normalised so the dataset-weighted mean is 1:

        sum(count[L] * w[L]) == sum(count[L])

    Labels with ``count == 0`` are **omitted** (no NaN, no inf, no
    division by zero). Determinism: input dict insertion order is
    preserved in the output.

    >>> w = compute_mode_weights({"rare": 100, "common": 100_000})
    >>> w["rare"] > w["common"]
    True
    >>> w = compute_mode_weights({"a": 10, "b": 10}, alpha=0.5)
    >>> abs(w["a"] - 1.0) < 1e-12 and abs(w["b"] - 1.0) < 1e-12
    True
    """
    if alpha < 0.0:
        msg = f"alpha must be >= 0, got {alpha}"
        raise ValueError(msg)
    nonzero = {k: int(v) for k, v in counts.items() if v > 0}
    if not nonzero:
        return {}
    raw: dict[str, float] = {label: 1.0 / (c**alpha) for label, c in nonzero.items()}
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


def boot_mode_weights(train_df: pl.DataFrame, *, alpha: float = _DEFAULT_ALPHA) -> pl.DataFrame:
    """Compute mode weights and attach them to *train_df*; emit boot log.

    Counts labels in ``train_df["fdm_mode_label"]``, computes power-law
    weights with exponent *alpha*, appends ``fdm_train_weight``, and logs
    one ``mode_weights_computed`` event carrying the diagnostic fields.
    """
    counts_series = train_df.get_column(_LABEL_COL).value_counts()
    counts: dict[str, int] = {
        row[0]: int(row[1]) for row in counts_series.iter_rows() if row[0] is not None
    }
    weights = compute_mode_weights(counts, alpha=alpha)
    out = attach_sample_weights(train_df, weights)
    nonzero_counts = {k: v for k, v in counts.items() if v > 0}
    nonzero_vals = list(nonzero_counts.values())
    imbalance = max(nonzero_vals) / min(nonzero_vals) if len(nonzero_vals) >= 2 else 1.0
    weight_vals = list(weights.values())
    log.info(
        "mode_weights_computed",
        alpha=alpha,
        imbalance_ratio=imbalance,
        n_labels=len(weights),
        top5_labels=_top_k(weights, counts, 5, descending=True),
        bottom5_labels=_top_k(weights, counts, 5, descending=False),
        weight_ratio_max_over_min=(max(weight_vals) / min(weight_vals) if weight_vals else 1.0),
    )
    return out
