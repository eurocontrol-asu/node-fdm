from __future__ import annotations

import math

import pytest
import torch

from node_fdm.layers.mass_encoder import MassEncoderLinear

FEATURE_COLS = [
    "dist_total_flight",
    "dist_adep_at_t0",
    "flight_time_remaining",
    "fuel_burned_so_far",
    "aircraft_weight_class",
]
EXPECTED_SIGNS = [1.0, -1.0, -1.0, -1.0, 1.0]
OEW_KG = 42600.0
MTOW_KG = 77000.0


def feature_stats() -> dict[str, dict[str, float]]:
    return {
        col: {"mean": 10.0 * (idx + 1), "std": 2.0 * (idx + 1)}
        for idx, col in enumerate(FEATURE_COLS)
    }


def build_encoder() -> MassEncoderLinear:
    return MassEncoderLinear(
        feature_stats=feature_stats(),
        feature_cols=FEATURE_COLS,
        expected_signs=EXPECTED_SIGNS,
        oew_kg=OEW_KG,
        mtow_kg=MTOW_KG,
    )


def mean_features(batch: int = 1) -> torch.Tensor:
    values = [feature_stats()[col]["mean"] for col in FEATURE_COLS]
    return torch.tensor([values] * batch, dtype=torch.float32)


def test_constructs_with_six_parameters_for_five_features() -> None:
    """AC1, AC9: public class exposes one coefficient per feature plus b0."""
    encoder = build_encoder()

    assert sum(p.numel() for p in encoder.parameters()) == 6


def test_constructor_rejects_signs_length_mismatch() -> None:
    """AC2: constructor rejects mismatched feature/sign lengths."""
    with pytest.raises(ValueError):
        MassEncoderLinear(
            feature_stats=feature_stats(),
            feature_cols=FEATURE_COLS,
            expected_signs=EXPECTED_SIGNS[:-1],
            oew_kg=OEW_KG,
            mtow_kg=MTOW_KG,
        )


def test_output_within_bounds_for_extreme_inputs() -> None:
    """AC6: finite extreme inputs remain inside configured mass bounds."""
    encoder = build_encoder()
    flight_features = torch.tensor([[1e6, -1e6, 1e6, -1e6, 1e6]], dtype=torch.float32)

    output = encoder.forward(flight_features)

    assert torch.all(output >= OEW_KG)
    assert torch.all(output <= MTOW_KG)


def test_output_at_neutral_init_close_to_b0_inverse() -> None:
    """AC4: neutral normalized input produces the sigmoid(b0_init) mass."""
    encoder = build_encoder()
    expected = OEW_KG + torch.sigmoid(torch.tensor(0.85)).item() * (MTOW_KG - OEW_KG)

    output = encoder.forward(mean_features())

    assert output.item() == pytest.approx(expected, abs=50.0)


def test_monotonic_in_positive_sign_feature() -> None:
    """AC7: positive-sign features make mass strictly increase."""
    encoder = build_encoder()
    stats = feature_stats()
    features = mean_features(batch=11)
    col = FEATURE_COLS[0]
    mean = stats[col]["mean"]
    std = stats[col]["std"]
    features[:, 0] = torch.linspace(mean - 3.0 * std, mean + 3.0 * std, 11)

    output = encoder.forward(features)

    assert torch.all(output[1:] > output[:-1])


def test_monotonic_in_negative_sign_feature() -> None:
    """AC7: negative-sign features make mass strictly decrease."""
    encoder = build_encoder()
    stats = feature_stats()
    features = mean_features(batch=11)
    col = FEATURE_COLS[1]
    mean = stats[col]["mean"]
    std = stats[col]["std"]
    features[:, 1] = torch.linspace(mean - 3.0 * std, mean + 3.0 * std, 11)

    output = encoder.forward(features)

    assert torch.all(output[1:] < output[:-1])


def test_effective_coefficients_returns_per_feature_dict() -> None:
    """AC8: diagnostics include b0 and one signed coefficient per feature."""
    encoder = build_encoder()

    coefficients = encoder.effective_coefficients()

    assert set(coefficients) == {"b0", *FEATURE_COLS}
    for col, expected_sign in zip(FEATURE_COLS, EXPECTED_SIGNS, strict=True):
        assert math.copysign(1.0, coefficients[col]) == expected_sign


def test_forward_batched_shape() -> None:
    """AC6: batched inputs return one mass per batch row."""
    encoder = build_encoder()
    features = torch.randn((8, 5), dtype=torch.float32)

    output = encoder.forward(features)

    assert output.shape == (8,)


def test_normalizer_uses_provided_stats() -> None:
    """AC3: forward uses provided mean/std statistics for normalization."""
    stats = feature_stats()
    encoder = MassEncoderLinear(
        feature_stats=stats,
        feature_cols=FEATURE_COLS,
        expected_signs=EXPECTED_SIGNS,
        oew_kg=OEW_KG,
        mtow_kg=MTOW_KG,
    )
    features = mean_features()
    features[0, 0] = stats[FEATURE_COLS[0]]["mean"] + stats[FEATURE_COLS[0]]["std"]
    expected_b = torch.nn.functional.softplus(torch.zeros(1)).item()
    expected_z = 0.85 + expected_b
    expected_mass = OEW_KG + torch.sigmoid(torch.tensor(expected_z)).item() * (MTOW_KG - OEW_KG)

    output = encoder.forward(features)

    assert output.item() == pytest.approx(expected_mass, abs=1e-3)
