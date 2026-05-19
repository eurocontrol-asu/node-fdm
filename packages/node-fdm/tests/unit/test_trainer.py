"""Unit tests for ``ODETrainer.identifiability_test_absolute`` (AXM-1741).

Covers AC1 (return-key contract, no-mass-encoder branch) and AC2 (sigma_obs
behavior — explicit override vs empirical computation from val_dataset dx).

Integration smoke tests live in
``tests/integration/test_identifiability_test_absolute.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch

import node_fdm.architectures.adsb
import node_fdm.architectures.adsb_hybrid  # noqa: F401  -- self-registers hybrid arch
from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import ODETrainer, TrainingConfig
from node_fdm_data.schemas.adsb_hybrid import FLIGHT_FEATURE_COLS_5 as FLIGHT_FEATURE_COLS

N_FEATURES = len(FLIGHT_FEATURE_COLS)

HYBRID_ALPHA: dict[str, float] = {
    "raw_alt_m": 1.0,
    "fdm_gamma_rad": 1.0,
    "era_tas_ms": 1.0,
    "fdm_heading_rad": 1.0,
    "fdm_mass_kg": 0.0,
}


def _make_hybrid_dataset(
    n_samples: int = 8,
    seq_len: int = 4,
) -> FlightDataset:
    """Tiny synthetic dataset compatible with the hybrid 5-D state."""
    torch.manual_seed(0)
    samples: list[FlightSample] = []
    n_u = 8
    n_e = 4
    for i in range(n_samples):
        jitter = 1.0 + 0.01 * i
        x0 = torch.tensor(
            [5000.0 * jitter, 0.01 * i, 200.0 * jitter, 0.01 * i, 66700.0 + 200.0 * i]
        )
        velocity = torch.tensor([0.5, 1e-4, 0.5, 1e-4, 0.0])
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        x = x0.unsqueeze(0) + t * velocity.unsqueeze(0)
        dx = velocity.unsqueeze(0).expand(seq_len, 5).clone()
        ff_template = [3_000_000.0 + 1e4 * i, 100_000.0, 11_000.0 + 50.0 * i, 5.0, 2.0, 0.78]
        ff_row = torch.tensor(ff_template[:N_FEATURES])
        flight_features = ff_row.unsqueeze(0).expand(seq_len, N_FEATURES).clone()
        samples.append(
            FlightSample(
                x=x,
                u=torch.zeros(seq_len, n_u),
                e=torch.zeros(seq_len, n_e),
                dx=dx,
                flight_features=flight_features,
            )
        )
    return FlightDataset(samples)


def _make_baseline_dataset(
    n_samples: int = 8,
    seq_len: int = 4,
) -> FlightDataset:
    """Tiny synthetic dataset compatible with the baseline 4-D ADS-B arch."""
    torch.manual_seed(0)
    samples: list[FlightSample] = []
    for i in range(n_samples):
        jitter = 1.0 + 0.01 * i
        x0 = torch.tensor([5000.0 * jitter, 0.01 * i, 200.0 * jitter, 0.01 * i])
        velocity = torch.tensor([0.5, 1e-4, 0.5, 1e-4])
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        x = x0.unsqueeze(0) + t * velocity.unsqueeze(0)
        dx = velocity.unsqueeze(0).expand(seq_len, 4).clone()
        samples.append(
            FlightSample(
                x=x,
                u=torch.zeros(seq_len, 8),
                e=torch.zeros(seq_len, 4),
                dx=dx,
            )
        )
    return FlightDataset(samples)


def _make_baseline_trainer(tmp_path: Path) -> ODETrainer:
    cfg = TrainingConfig(
        architecture_name="node_adsb_v1",
        model_name="baseline_absid",
        epochs=1,
        batch_size=4,
        num_workers=0,
        val_batch_size=4,
        seq_len=4,
        step=1.0,
        method="euler",
        seed=0,
    )
    return ODETrainer(
        config=cfg,
        train_dataset=_make_baseline_dataset(),
        val_dataset=_make_baseline_dataset(n_samples=4),
        model_dir=tmp_path,
    )


def _make_hybrid_trainer(tmp_path: Path) -> ODETrainer:
    cfg = TrainingConfig(
        architecture_name="node_adsb_hybrid_v1",
        model_name="hybrid_absid",
        epochs=1,
        batch_size=4,
        num_workers=0,
        val_batch_size=4,
        seq_len=4,
        step=1.0,
        method="euler",
        lr=1e-3,
        alpha_dict=HYBRID_ALPHA,
        seed=0,
    )
    return ODETrainer(
        config=cfg,
        train_dataset=_make_hybrid_dataset(),
        val_dataset=_make_hybrid_dataset(n_samples=4),
        model_dir=tmp_path,
    )


def test_identifiability_test_absolute_returns_zero_score_for_baseline_without_mass_encoder(
    tmp_path: Path,
) -> None:
    """AC1: baseline trainer (no MassEncoder) returns zeroed-out absolute score."""
    trainer = _make_baseline_trainer(tmp_path)
    assert trainer.mass_encoder is None

    result = trainer.identifiability_test_absolute(factor=1.3)

    assert result["delta_abs"] == 0.0
    assert result["absolute_score"] == 0.0
    assert result["passed_absolute"] is False


def test_identifiability_test_absolute_returns_all_required_keys(
    tmp_path: Path,
) -> None:
    """AC1: hybrid trainer returns dict with all six required keys."""
    trainer = _make_hybrid_trainer(tmp_path)

    result = trainer.identifiability_test_absolute(factor=1.3)

    expected = {
        "baseline_mse",
        "perturbed_mse",
        "delta_abs",
        "sigma_obs_sq",
        "absolute_score",
        "passed_absolute",
    }
    assert expected.issubset(set(result.keys()))


def test_identifiability_test_absolute_uses_explicit_sigma_obs_when_provided(
    tmp_path: Path,
) -> None:
    """AC2: explicit sigma_obs is used verbatim (not recomputed empirically)."""
    trainer = _make_hybrid_trainer(tmp_path)
    explicit_sigma_sq = 42.0

    result = trainer.identifiability_test_absolute(factor=1.3, sigma_obs=explicit_sigma_sq)

    assert result["sigma_obs_sq"] == explicit_sigma_sq


def test_identifiability_test_absolute_computes_empirical_sigma_obs_from_val_dx(
    tmp_path: Path,
) -> None:
    """AC2: when sigma_obs is None, sigma_obs_sq = Σ alpha_i · var(dx_i) on val."""
    trainer = _make_hybrid_trainer(tmp_path)

    # Recompute the expected weighted variance from the val dataset's dx.
    dx_stack = torch.stack([trainer.val_dataset[i].dx for i in range(len(trainer.val_dataset))])
    # Flatten (n_samples, seq_len, n_dx) -> (n_samples*seq_len, n_dx).
    dx_flat = dx_stack.reshape(-1, dx_stack.shape[-1])
    per_col_var = dx_flat.var(dim=0, unbiased=False)
    alpha = trainer._alpha_weights
    # spec.dx_cols has the same length as alpha; broadcast on the dx axis.
    expected_sigma_sq = float((alpha * per_col_var).sum().item())

    result = trainer.identifiability_test_absolute(factor=1.3)

    assert result["sigma_obs_sq"] == expected_sigma_sq
