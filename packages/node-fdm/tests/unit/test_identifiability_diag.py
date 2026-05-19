"""Unit tests for the ``scripts/identifiability_diag.py`` helper (AXM-1741, AC6).

Verifies the two reusable functions consumed by
``axm1740_recalibrate.py``, ``phase15_parity_report.py`` and the future
``phase2_drag_report.py``: empirical sigma_obs computation and the markdown
formatter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from node_fdm.dataset import FlightDataset, FlightSample

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.identifiability_diag import (  # type: ignore[import-not-found]  # noqa: E402
    compute_sigma_obs,
    format_identifiability_section,
)

DX_COLS = ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms", "fdm_heading_rad"]


def _make_val_dataset(
    n_samples: int = 6,
    seq_len: int = 4,
    n_dx: int = 4,
) -> FlightDataset:
    """Build a tiny FlightDataset where dx has a known non-zero variance."""
    torch.manual_seed(7)
    samples: list[FlightSample] = []
    for i in range(n_samples):
        x = torch.zeros(seq_len, n_dx)
        dx = torch.randn(seq_len, n_dx) + float(i)
        samples.append(
            FlightSample(
                x=x,
                u=torch.zeros(seq_len, 8),
                e=torch.zeros(seq_len, 4),
                dx=dx,
            )
        )
    return FlightDataset(samples)


def test_compute_sigma_obs_returns_per_dx_std() -> None:
    """AC6: ``compute_sigma_obs`` returns a dict keyed by dx_col with empirical std."""
    val_dataset = _make_val_dataset()

    result = compute_sigma_obs(val_dataset, DX_COLS)

    assert set(result.keys()) == set(DX_COLS)
    dx_stack = torch.stack(
        [val_dataset[i].dx for i in range(len(val_dataset))],
    ).reshape(-1, len(DX_COLS))
    expected_stds = dx_stack.std(dim=0, unbiased=False)
    for idx, col in enumerate(DX_COLS):
        assert result[col] == pytest.approx(expected_stds[idx].item(), rel=1e-5)


def test_compute_sigma_obs_raises_on_empty_dataset() -> None:
    """AC6: ``compute_sigma_obs`` raises ``ValueError`` when no dx values are available.

    A ``FlightDataset`` rejects an empty sample list at construction, so we
    exercise the helper's own guard via an empty ``dx_cols`` argument — the
    helper must not return an empty dict silently when its caller passes
    no columns to measure.
    """
    val_dataset = _make_val_dataset()

    with pytest.raises(ValueError):
        compute_sigma_obs(val_dataset, [])


def test_format_identifiability_section_includes_required_fields() -> None:
    """AC6: formatter emits baseline_mse, perturbed_mse, absolute_score, k_min, verdict."""
    result = {
        "baseline_mse": 0.123456,
        "perturbed_mse": 0.345678,
        "delta_abs": 0.222222,
        "sigma_obs_sq": 1.5,
        "absolute_score": 0.148148,
        "passed_absolute": True,
    }

    section = format_identifiability_section(result, model_name="full_hybrid_v2", k_min=0.1)

    assert "baseline_mse" in section
    assert "perturbed_mse" in section
    assert "absolute_score" in section
    assert "k_min" in section
    assert ("PASS" in section) or ("FAIL" in section)


def test_format_identifiability_section_omits_k_min_when_none() -> None:
    """AC6: with k_min=None the formatter switches to calibration-mode verdict."""
    result = {
        "baseline_mse": 0.1,
        "perturbed_mse": 0.2,
        "delta_abs": 0.1,
        "sigma_obs_sq": 1.0,
        "absolute_score": 0.1,
        "passed_absolute": False,
    }

    section = format_identifiability_section(result, model_name="full_hybrid_v2", k_min=None)

    # No explicit k_min line — the field must not appear as a labelled row.
    assert "k_min:" not in section
    assert "k_min =" not in section
    assert "n/a (calibration mode)" in section
