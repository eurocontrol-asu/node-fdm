from __future__ import annotations

import math

import polars as pl
import pytest

from node_fdm.trainer import TrainingConfig
from node_fdm.training.weighting import (
    attach_sample_weights,
    auto_beta,
    compute_mode_weights,
)

# ---- auto_beta -------------------------------------------------------------


def test_auto_beta_in_clamp_range_for_realistic_imbalance() -> None:
    counts = {"A": 1_000_000, "B": 1_000}
    beta = auto_beta(counts)
    expected = max(0.99, min(0.9999, 1 - 1 / math.sqrt(1000)))
    assert 0.99 <= beta <= 0.9999
    assert math.isclose(beta, expected, abs_tol=1e-9)


def test_auto_beta_clamps_to_lower_bound_when_ratio_small() -> None:
    assert auto_beta({"A": 100, "B": 90}) == 0.99


def test_auto_beta_clamps_to_upper_bound_when_ratio_huge() -> None:
    assert auto_beta({"A": 10**12, "B": 1}) == 0.9999


def test_auto_beta_raises_on_zero_count() -> None:
    with pytest.raises(ValueError):
        auto_beta({"A": 1000, "B": 0})


def test_auto_beta_raises_on_single_label() -> None:
    with pytest.raises(ValueError):
        auto_beta({"A": 1000})


# ---- compute_mode_weights --------------------------------------------------


_REPRESENTATIVE_COUNTS = {
    "TURN": 100,
    "ALT_MACH": 520_000,
    "ALT_CAS": 300_000,
    "VS_CAS": 1_500,
    "VS_MACH": 9_000,
    "GAMMA_CAS": 600,
    "GAMMA_MACH": 4_000,
    "FPA_CAS": 300,
    "FPA_MACH": 200,
    "LEVEL": 250_000,
    "CLIMB": 180_000,
    "DESCENT": 200_000,
    "GROUND": 1_000,
}


def test_compute_mode_weights_dataset_weighted_mean_is_one() -> None:
    weights = compute_mode_weights(_REPRESENTATIVE_COUNTS)
    total = sum(_REPRESENTATIVE_COUNTS.values())
    weighted_sum = sum(_REPRESENTATIVE_COUNTS[k] * weights[k] for k in _REPRESENTATIVE_COUNTS)
    assert math.isclose(weighted_sum, total, rel_tol=1e-6, abs_tol=1e-6)


def test_compute_mode_weights_rare_modes_get_higher_weight_than_dominant() -> None:
    weights = compute_mode_weights(_REPRESENTATIVE_COUNTS)
    assert weights["TURN"] > weights["ALT_MACH"]


def test_compute_mode_weights_strictly_positive() -> None:
    weights = compute_mode_weights(_REPRESENTATIVE_COUNTS)
    assert all(w > 0 for w in weights.values())


def test_compute_mode_weights_omits_zero_count_labels() -> None:
    counts = dict(_REPRESENTATIVE_COUNTS)
    counts["GAMMA_CAS"] = 0
    weights = compute_mode_weights(counts)
    assert "GAMMA_CAS" not in weights
    assert all(math.isfinite(w) for w in weights.values())


def test_compute_mode_weights_deterministic_on_repeated_call() -> None:
    counts = dict(_REPRESENTATIVE_COUNTS)
    a = compute_mode_weights(counts)
    b = compute_mode_weights(counts)
    assert list(a.keys()) == list(b.keys())
    assert a == b


# ---- attach_sample_weights -------------------------------------------------


def test_attach_sample_weights_adds_float32_column() -> None:
    df = pl.DataFrame({"fdm_mode_label": ["TURN", "ALT_MACH", "TURN", "ALT_MACH"]})
    weights = {"TURN": 5.0, "ALT_MACH": 0.5}
    out = attach_sample_weights(df, weights)
    assert out.schema["fdm_train_weight"] == pl.Float32
    assert out.get_column("fdm_train_weight").to_list() == [5.0, 0.5, 5.0, 0.5]


def test_attach_sample_weights_raises_on_unknown_label() -> None:
    df = pl.DataFrame({"fdm_mode_label": ["NOT_IN_WEIGHTS"]})
    with pytest.raises(KeyError, match="NOT_IN_WEIGHTS"):
        attach_sample_weights(df, {"TURN": 1.0})


# ---- TrainingConfig --------------------------------------------------------


def test_training_config_defaults_use_mode_weights_to_false() -> None:
    cfg = TrainingConfig(architecture_name="node_adsb_v1", model_name="m")
    assert cfg.use_mode_weights is False


def test_training_config_accepts_use_mode_weights_true() -> None:
    cfg = TrainingConfig(architecture_name="node_adsb_v1", model_name="m", use_mode_weights=True)
    dumped = cfg.model_dump()
    assert dumped["use_mode_weights"] is True
