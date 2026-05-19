"""Integration tests for ``ODETrainer.identifiability_test_absolute`` (AXM-1741).

Covers AC1 + AC2 (smoke on a trained-ish hybrid MassEncoder, end-to-end
through ``_compute_batch_loss`` on a real val_dataset) and AC7
(non-regression of the legacy ``identifiability_test`` signature).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import node_fdm.architectures.adsb_hybrid  # noqa: F401  -- self-registers hybrid arch
from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import ODETrainer, TrainingConfig
from node_fdm_data.schemas.adsb_hybrid import FLIGHT_FEATURE_COLS_5 as FLIGHT_FEATURE_COLS

pytestmark = pytest.mark.integration

N_FEATURES = len(FLIGHT_FEATURE_COLS)

HYBRID_ALPHA: dict[str, float] = {
    "raw_alt_m": 1.0,
    "fdm_gamma_rad": 1.0,
    "era_tas_ms": 1.0,
    "fdm_heading_rad": 1.0,
    "fdm_mass_kg": 0.0,
}


def _make_hybrid_dataset(
    n_samples: int = 12,
    seq_len: int = 4,
) -> FlightDataset:
    """Tiny synthetic dataset matching the hybrid 5-D state shape.

    Per-sample dx variance ensures ``sigma_obs_sq > 0`` (alpha-weighted
    variance over the val set). The flight features carry the
    per-sample signal that the MassEncoder learns, so a 1.3x perturbation
    on ``m_0`` shifts ``(T-D)/m`` out of distribution and degrades the
    val MSE once the model has trained enough epochs.
    """
    torch.manual_seed(0)
    samples: list[FlightSample] = []
    n_u = 8
    n_e = 4
    for i in range(n_samples):
        jitter = 1.0 + 0.01 * i
        true_mass = 66700.0 + 200.0 * i
        x0 = torch.tensor([5000.0 * jitter, 0.01 * i, 200.0 * jitter, 0.01 * i, true_mass])
        # Per-sample dx so var(dx_i) > 0 across the val set — needed for
        # sigma_obs_sq to be strictly positive (alpha-weighted variance).
        velocity = torch.tensor(
            [0.5 + 0.05 * i, 1e-4 + 1e-5 * i, 0.5 + 0.05 * i, 1e-4 + 1e-5 * i, 0.0]
        )
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


def _make_hybrid_trainer(tmp_path: Path, *, epochs: int = 60) -> ODETrainer:
    cfg = TrainingConfig(
        architecture_name="node_adsb_hybrid_v1",
        model_name="hybrid_absid_integration",
        epochs=epochs,
        batch_size=4,
        num_workers=0,
        val_batch_size=4,
        seq_len=4,
        step=1.0,
        method="euler",
        lr=1e-2,
        alpha_dict=HYBRID_ALPHA,
        seed=0,
    )
    return ODETrainer(
        config=cfg,
        train_dataset=_make_hybrid_dataset(),
        val_dataset=_make_hybrid_dataset(n_samples=4),
        model_dir=tmp_path,
    )


def test_identifiability_test_absolute_runs_on_hybrid_v2_smoke(tmp_path: Path) -> None:
    """AC1, AC2: trained hybrid trainer returns a valid dict and a positive score.

    The trainer is briefly trained so the MassEncoder departs from its init,
    making the 1.3x perturbation push ``(T-D)/m`` out of distribution and
    increase the val MSE — ``absolute_score`` should be strictly positive.
    """
    trainer = _make_hybrid_trainer(tmp_path, epochs=60)
    trainer.train()

    result = trainer.identifiability_test_absolute(factor=1.3)

    assert set(result.keys()) >= {
        "baseline_mse",
        "perturbed_mse",
        "delta_abs",
        "sigma_obs_sq",
        "absolute_score",
        "passed_absolute",
    }
    assert math.isfinite(result["baseline_mse"]) and result["baseline_mse"] > 0
    assert math.isfinite(result["perturbed_mse"]) and result["perturbed_mse"] > 0
    assert math.isfinite(result["sigma_obs_sq"]) and result["sigma_obs_sq"] > 0
    assert result["absolute_score"] > 0


def test_identifiability_test_absolute_preserves_legacy_method(tmp_path: Path) -> None:
    """AC7: ``identifiability_test`` still returns its legacy 3-key dict, unchanged."""
    trainer = _make_hybrid_trainer(tmp_path, epochs=2)
    trainer.train()

    result = trainer.identifiability_test(factor=1.3)

    assert set(result.keys()) == {"baseline_mse", "perturbed_mse", "ratio"}
