"""Unit tests for MassEncoder/hybrid wiring in the trainer.

Covers the slices of AC1, AC2, AC5, AC7, AC10 that do not require a full
training loop or disk round-trip. Integration-level smoke tests live in
``tests/integration/test_trainer_hybrid_smoke.py``.
"""

from __future__ import annotations

from pathlib import Path

import structlog.testing
import torch

import node_fdm.architectures.adsb_hybrid  # noqa: F401  -- self-registers hybrid arch
from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.layers.mass_encoder import MassEncoderLinear
from node_fdm.trainer import ODETrainer, TrainingConfig, _collate_flight_samples

FLIGHT_FEATURE_COLS = [
    "dist_total_flight",
    "dist_adep_at_t0",
    "cruise_alt_max_flight",
    "wind_long_mean_flight",
    "temp_isa_dev_mean_flight",
]

N_FEATURES = len(FLIGHT_FEATURE_COLS)


def _make_hybrid_dataset(
    n_samples: int = 10,
    seq_len: int = 4,
) -> FlightDataset:
    """Build a tiny synthetic dataset compatible with the hybrid arch.

    The hybrid arch has 5 state dims (raw_alt_m, fdm_gamma_rad, era_tas_ms,
    fdm_heading_rad, fdm_mass_kg), 5 ODE controls, ``E0_COLS`` and ``E1_COLS``
    matching the baseline, and one entry per flight feature.
    """
    torch.manual_seed(0)
    samples: list[FlightSample] = []
    n_x = 5
    n_u = 8
    n_e = 4
    for _ in range(n_samples):
        x0 = torch.tensor([5000.0, 0.0, 200.0, 0.0, 60000.0])
        x_step = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        x = x0.unsqueeze(0) + t * x_step.unsqueeze(0)
        dx = x_step.unsqueeze(0).expand(seq_len, n_x)
        ff_row = torch.tensor([3_000_000.0, 100_000.0, 11_000.0, 5.0, 2.0])
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
    """Build a tiny synthetic dataset compatible with the baseline ADS-B arch."""
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


def _make_hybrid_trainer(
    tmp_path: Path,
    *,
    alpha_dict: dict[str, float] | None = None,
) -> ODETrainer:
    """Build a hybrid ``ODETrainer`` on the tiny synthetic dataset."""
    default_alpha: dict[str, float] = {
        "raw_alt_m": 1.0,
        "fdm_gamma_rad": 1.0,
        "era_tas_ms": 1.0,
        "fdm_heading_rad": 1.0,
        "fdm_mass_kg": 0.0,
    }
    cfg = TrainingConfig(
        architecture_name="node_adsb_hybrid_v1",
        model_name="hybrid_unit",
        epochs=1,
        batch_size=4,
        num_workers=0,
        val_batch_size=4,
        seq_len=4,
        step=1.0,
        method="euler",
        alpha_dict=alpha_dict if alpha_dict is not None else default_alpha,
    )
    return ODETrainer(
        config=cfg,
        train_dataset=_make_hybrid_dataset(),
        val_dataset=_make_hybrid_dataset(n_samples=4),
        model_dir=tmp_path,
    )


def test_collate_includes_flight_features(tmp_path: Path) -> None:
    """AC5: ``_collate_flight_samples`` appends a stacked ``flight_features`` tensor."""
    seq_len = 6
    n_features = 5
    samples = [
        FlightSample(
            x=torch.zeros(seq_len, 5),
            u=torch.zeros(seq_len, 8),
            e=torch.zeros(seq_len, 4),
            dx=torch.zeros(seq_len, 5),
            flight_features=torch.ones(seq_len, n_features) * float(i),
        )
        for i in range(3)
    ]

    out = _collate_flight_samples(samples)

    assert out[-1].shape == (3, seq_len, n_features)
    assert torch.equal(out[-1][0], torch.zeros(seq_len, n_features))
    assert torch.equal(out[-1][2], torch.ones(seq_len, n_features) * 2.0)


def test_mass_encoder_instantiated_for_hybrid_arch(tmp_path: Path) -> None:
    """AC1: hybrid arch gives the trainer a ``MassEncoderLinear`` instance."""
    trainer = _make_hybrid_trainer(tmp_path)

    assert trainer.mass_encoder is not None
    assert isinstance(trainer.mass_encoder, MassEncoderLinear)


def test_mass_encoder_none_for_baseline_arch(tmp_path: Path) -> None:
    """AC10: baseline arch leaves ``trainer.mass_encoder`` at ``None``."""
    cfg = TrainingConfig(
        architecture_name="node_adsb_v1",
        model_name="baseline_unit",
        epochs=1,
        batch_size=4,
        num_workers=0,
        val_batch_size=4,
        seq_len=4,
        step=1.0,
        method="euler",
    )
    trainer = ODETrainer(
        config=cfg,
        train_dataset=_make_baseline_dataset(),
        val_dataset=_make_baseline_dataset(n_samples=4),
        model_dir=tmp_path,
    )

    assert trainer.mass_encoder is None


def test_optimizer_includes_mass_encoder_params(tmp_path: Path) -> None:
    """AC2: optimizer's first param group covers the 6 mass-encoder parameters."""
    trainer = _make_hybrid_trainer(tmp_path)

    assert trainer.mass_encoder is not None
    optim_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    mass_param_ids = {id(p) for p in trainer.mass_encoder.parameters()}

    assert mass_param_ids.issubset(optim_param_ids)
    assert sum(p.numel() for p in trainer.mass_encoder.parameters()) == 6


def test_alpha_dict_warns_when_mass_unweighted(tmp_path: Path) -> None:
    """AC7: missing ``fdm_mass_kg`` from a non-empty ``alpha_dict`` emits a warning."""
    with structlog.testing.capture_logs() as captured:
        _make_hybrid_trainer(tmp_path, alpha_dict={"raw_alt_m": 1.0})

    events = [entry.get("event") for entry in captured]
    assert "mass_dim_unweighted" in events, (
        f"Expected a 'mass_dim_unweighted' warning, got events: {events}"
    )
