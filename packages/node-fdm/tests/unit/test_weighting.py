from __future__ import annotations

import math

import polars as pl
import pytest

from node_fdm.trainer import TrainingConfig
from node_fdm.training.weighting import (
    attach_sample_weights,
    compute_mode_weights,
)

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


def test_compute_mode_weights_alpha_zero_is_uniform() -> None:
    weights = compute_mode_weights(_REPRESENTATIVE_COUNTS, alpha=0.0)
    assert all(math.isclose(w, 1.0, abs_tol=1e-12) for w in weights.values())


def test_compute_mode_weights_alpha_one_recovers_inverse_frequency_shape() -> None:
    weights = compute_mode_weights(_REPRESENTATIVE_COUNTS, alpha=1.0)
    # With alpha=1 the unnormalised weight is 1/count, so the *product*
    # count * w is the same constant for every label (up to normalisation).
    counts = _REPRESENTATIVE_COUNTS
    products = [counts[k] * weights[k] for k in counts]
    assert all(math.isclose(p, products[0], rel_tol=1e-9) for p in products)


def test_compute_mode_weights_higher_alpha_boosts_rare_more() -> None:
    counts = {"rare": 100, "common": 100_000}
    w_low = compute_mode_weights(counts, alpha=0.25)
    w_high = compute_mode_weights(counts, alpha=0.75)
    ratio_low = w_low["rare"] / w_low["common"]
    ratio_high = w_high["rare"] / w_high["common"]
    assert ratio_high > ratio_low


def test_compute_mode_weights_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        compute_mode_weights(_REPRESENTATIVE_COUNTS, alpha=-0.1)


def test_compute_mode_weights_handles_single_label() -> None:
    weights = compute_mode_weights({"only": 1000})
    assert math.isclose(weights["only"], 1.0, abs_tol=1e-12)


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


def test_training_config_default_mode_weight_alpha_is_half() -> None:
    cfg = TrainingConfig(architecture_name="node_adsb_v1", model_name="m")
    assert cfg.mode_weight_alpha == 0.5


def test_training_config_rejects_alpha_outside_unit_interval() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrainingConfig(architecture_name="node_adsb_v1", model_name="m", mode_weight_alpha=1.5)
    with pytest.raises(ValidationError):
        TrainingConfig(architecture_name="node_adsb_v1", model_name="m", mode_weight_alpha=-0.1)
