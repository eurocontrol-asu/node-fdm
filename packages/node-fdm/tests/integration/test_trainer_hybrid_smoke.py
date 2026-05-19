"""Integration smoke tests for the hybrid (MassEncoder) training loop.

These tests train ``ODETrainer`` end-to-end on a tiny synthetic dataset for
two epochs, then check the identifiability gate, the post-train coefficient
logger and the save/load round-trip. Pure config / collate slices live in
``tests/unit/test_trainer_hybrid_smoke.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import node_fdm.architectures.adsb_hybrid  # noqa: F401  -- self-registers hybrid arch
from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import ODETrainer, TrainingConfig

pytestmark = pytest.mark.integration

FLIGHT_FEATURE_COLS = [
    "dist_total_flight",
    "dist_adep_at_t0",
    "cruise_alt_max_flight",
    "wind_long_mean_flight",
    "temp_isa_dev_mean_flight",
]

N_FEATURES = len(FLIGHT_FEATURE_COLS)

HYBRID_ALPHA: dict[str, float] = {
    "raw_alt_m": 1.0,
    "fdm_gamma_rad": 1.0,
    "era_tas_ms": 1.0,
    "fdm_heading_rad": 1.0,
    "fdm_mass_kg": 0.0,
}


def _make_hybrid_dataset(
    n_samples: int = 10,
    seq_len: int = 4,
) -> FlightDataset:
    """Tiny synthetic dataset matching the hybrid 5-D state shape."""
    torch.manual_seed(0)
    samples: list[FlightSample] = []
    n_x = 5
    n_u = 8
    n_e = 4
    for i in range(n_samples):
        # Tiny per-sample jitter so compute_stats produces a non-degenerate std
        # on every column (zero-std columns would cause NaN normalization).
        # The MassEncoder initial output is ~66 700 kg (sigmoid(b0_init=0.85)
        # mapped onto the [OEW, MTOW] = [42 600, 77 000] range). Pick the true
        # mass close to that bias so a small amount of training (20 epochs)
        # gives a backbone calibrated for m_0 ~ 66 700, and the 1.3
        # perturbation -- which saturates against the MTOW ceiling -- pushes
        # the per-step acceleration ``(T-D)/m`` out of distribution and
        # increases val MSE.
        jitter = 1.0 + 0.01 * i
        true_mass = 66700.0 + 200.0 * i
        x0 = torch.tensor([5000.0 * jitter, 0.01 * i, 200.0 * jitter, 0.01 * i, true_mass])
        # Non-trivial dx/dt on the longitudinal channel so the trained
        # backbone learns ``(T-D)/m`` calibrated for the true mass; a 30 %
        # mass perturbation then shifts the predicted acceleration away from
        # the value the backbone learned.
        velocity = torch.tensor([0.5, 1e-4, 0.5, 1e-4, 0.0])
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        x = x0.unsqueeze(0) + t * velocity.unsqueeze(0)
        dx = velocity.unsqueeze(0).expand(seq_len, n_x).clone()
        ff_row = torch.tensor([3_000_000.0 + 1e4 * i, 100_000.0, 11_000.0 + 50.0 * i, 5.0, 2.0])
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
    n_samples: int = 10,
    seq_len: int = 4,
) -> FlightDataset:
    """Tiny synthetic dataset matching the baseline 4-D state shape."""
    torch.manual_seed(0)
    samples = [
        FlightSample(
            x=torch.zeros(seq_len, 4),
            u=torch.zeros(seq_len, 8),
            e=torch.zeros(seq_len, 4),
            dx=torch.zeros(seq_len, 4),
        )
        for _ in range(n_samples)
    ]
    return FlightDataset(samples)


def _make_hybrid_trainer(tmp_path: Path, *, epochs: int = 2) -> ODETrainer:
    cfg = TrainingConfig(
        architecture_name="node_adsb_hybrid_v1",
        model_name="hybrid_smoke",
        epochs=epochs,
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


def _make_baseline_trainer(tmp_path: Path, *, epochs: int = 1) -> ODETrainer:
    cfg = TrainingConfig(
        architecture_name="node_adsb_v1",
        model_name="baseline_smoke",
        epochs=epochs,
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


def test_train_two_epochs_no_nan(tmp_path: Path) -> None:
    """AC13: two epochs on the tiny hybrid dataset run without NaN/Inf loss."""
    trainer = _make_hybrid_trainer(tmp_path, epochs=2)

    records = trainer.train()

    assert len(records) == 2
    final = records[-1]["train_loss"]
    assert final == final, f"train_loss is NaN: {final}"
    assert final != float("inf") and final != float("-inf")


def test_effective_coefficients_callable_post_train(tmp_path: Path) -> None:
    """AC8, AC13: ``mass_encoder.effective_coefficients()`` returns ``b0`` + 5 features."""
    trainer = _make_hybrid_trainer(tmp_path, epochs=2)
    trainer.train()

    assert trainer.mass_encoder is not None
    coefs = trainer.mass_encoder.effective_coefficients()

    assert "b0" in coefs
    for col in FLIGHT_FEATURE_COLS:
        assert col in coefs, f"Missing coefficient for feature {col!r}: {coefs}"


def test_identifiability_factor_degrades_val_mse(tmp_path: Path) -> None:
    """AC9, AC13: a ``factor=1.3`` perturbation degrades val MSE (``ratio > 1.0``).

    The default hybrid ``alpha_dict`` sets ``fdm_mass_kg: 0.0`` (mass residual
    silenced because ``dm/dt=0`` is enforced by ``dx_bounds``). For the
    identifiability gate we re-weight the mass dim so the perturbation on
    ``m_0`` shows up directly in the residual: at baseline the predicted
    mass tracks the true mass through the projector; at perturbed it sits at
    ``m_0 * 1.3`` (possibly clamped) and the residual jumps.
    """
    alpha = {
        "raw_alt_m": 1.0,
        "fdm_gamma_rad": 1.0,
        "era_tas_ms": 1.0,
        "fdm_heading_rad": 1.0,
        "fdm_mass_kg": 1.0,
    }
    cfg = TrainingConfig(
        architecture_name="node_adsb_hybrid_v1",
        model_name="hybrid_ident",
        epochs=10,
        batch_size=4,
        num_workers=0,
        val_batch_size=4,
        seq_len=4,
        step=1.0,
        method="euler",
        lr=1e-3,
        alpha_dict=alpha,
        seed=0,
    )
    trainer = ODETrainer(
        config=cfg,
        train_dataset=_make_hybrid_dataset(),
        val_dataset=_make_hybrid_dataset(n_samples=4),
        model_dir=tmp_path,
    )
    trainer.train()

    result = trainer.identifiability_test(factor=1.3)

    assert "ratio" in result
    assert result["ratio"] > 1.0, (
        f"Expected ratio > 1.0 (perturbation should hurt), got {result!r}"
    )


def test_identifiability_test_returns_unit_ratio_for_baseline(tmp_path: Path) -> None:
    """AC9: baseline arch (no MassEncoder) reports baseline == perturbed."""
    trainer = _make_baseline_trainer(tmp_path, epochs=1)
    trainer.train()

    result = trainer.identifiability_test(factor=1.3)

    assert result["baseline_mse"] == result["perturbed_mse"]


def test_save_meta_records_mass_encoder_flag(tmp_path: Path) -> None:
    """AC12: ``save_meta`` writes ``mass_encoder: true`` and a ``mass_encoder.pt`` file."""
    trainer = _make_hybrid_trainer(tmp_path, epochs=1)
    trainer.train()
    trainer.save_model(0)

    meta_path = trainer.model_dir / "meta.json"
    meta = json.loads(meta_path.read_text())

    assert meta.get("mass_encoder") is True
    assert (trainer.model_dir / "mass_encoder.pt").exists()


def test_load_model_weights_restores_mass_encoder(tmp_path: Path) -> None:
    """AC12: ``load_model_weights`` restores the saved MassEncoder ``b0`` parameter."""
    trainer1 = _make_hybrid_trainer(tmp_path, epochs=1)
    trainer1.train()
    trainer1.save_model(0)

    assert trainer1.mass_encoder is not None
    saved_b0 = trainer1.mass_encoder.b0.detach().clone()

    trainer2 = _make_hybrid_trainer(tmp_path, epochs=1)
    assert trainer2.mass_encoder is not None
    # Sanity: a fresh trainer's b0 is at the constant init, not the trained value.
    trainer2.mass_encoder.b0.data.fill_(0.0)

    trainer2.load_model_weights()

    assert torch.allclose(trainer2.mass_encoder.b0.detach(), saved_b0)
