"""Smoke test: ``node_adsb_v1`` (Phase 2B) instantiates and forward-passes.

Catches schema/col_map/input_cols mismatches in < 5 s by building the
full ``FlightDynamicsModel`` from synthetic stats and running one ODE
step on a synthetic batch.  Does NOT assert physical correctness — only
that shapes, dtypes and dictionary keys line up.
"""

from __future__ import annotations

import math

import torch

import node_fdm.architectures.adsb  # noqa: F401  (registers node_adsb_v1)
from node_fdm.architectures.registry import get
from node_fdm.models.fdm import FlightDynamicsModel


def _synthetic_stats(spec) -> dict[str, dict[str, float]]:  # type: ignore[no-untyped-def]
    """Provide mean/std/max/p999 for every column the model touches."""
    cols: set[str] = set(spec.x_cols) | set(spec.u_cols) | set(spec.e0_cols)
    cols |= set(spec.e1_cols)
    cols |= {col for _, col in spec.dx_cols}
    # NN outputs (StructuredLayer) need stats too — its denormaliser
    # consults stats_dict for the ``scaled`` path even though the
    # cap/scale overrides take precedence.
    for layer in spec.layers:
        cols |= set(layer.input_cols) | set(layer.output_cols)
    return {c: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 1.0, "iqr": 1.0} for c in cols}


def test_node_adsb_v1_forward_pass() -> None:
    """Build the model and run one forward step end-to-end."""
    spec = get("node_adsb_v1")
    stats = _synthetic_stats(spec)

    model = FlightDynamicsModel(spec, stats, model_params=(1, 1, 8))

    batch = 2
    n_x = len(spec.x_cols)
    n_u = len(spec.u_cols)
    n_e = len(spec.e0_cols)

    # Realistic-ish synthetic state: alt=5000m, gamma=0, tas=200, heading=π/2.
    x = torch.zeros((batch, n_x), dtype=torch.float32)
    x[:, spec.x_cols.index("raw_alt_m")] = 5000.0
    x[:, spec.x_cols.index("era_tas_ms")] = 200.0
    x[:, spec.x_cols.index("fdm_heading_rad")] = math.pi / 2

    u = torch.zeros((batch, n_u), dtype=torch.float32)
    # Mark the known-flags as 1 so the diff signals are non-zero (exercise
    # the masked path inside the TrajectoryLayer).
    u[:, spec.u_cols.index("fdm_gamma_target_known")] = 1.0
    u[:, spec.u_cols.index("fdm_tas_target_known")] = 1.0
    u[:, spec.u_cols.index("fdm_heading_target_known")] = 1.0
    u[:, spec.u_cols.index("fdm_heading_known")] = 1.0
    u[:, spec.u_cols.index("fdm_heading_target_rad")] = math.pi / 2 + 0.05

    e = torch.zeros((batch, n_e), dtype=torch.float32)
    e[:, spec.e0_cols.index("era_temp_K")] = 250.0
    e[:, spec.e0_cols.index("era_u_wind_ms")] = 5.0
    e[:, spec.e0_cols.index("era_v_wind_ms")] = -3.0

    dx = model.forward(x, u, e)

    # Shape check: (batch, n_dx).
    assert dx.shape == (batch, len(spec.dx_cols))
    assert torch.isfinite(dx).all(), f"non-finite dx: {dx}"

    # Spot-check the lateral derivative actually populated.
    d_heading_idx = [name for _, name in spec.dx_cols].index("fdm_d_heading_rads")
    d_heading = dx[:, d_heading_idx]
    # phi_bank is the NN output; with random init + cap=1.0 + tas=200 the
    # rate is bounded by g/V·tan(1) ≈ 9.81/200·1.557 ≈ 0.076 rad/s.
    assert torch.all(d_heading.abs() <= 0.077), f"d_heading out of bounds: {d_heading}"

    # And the lateral derived signals were stashed in history.
    for col in (
        "fdm_lat_wind_ms",
        "fdm_drift_rad",
        "fdm_track_rad",
        "fdm_heading_diff_rad",
    ):
        assert col in model.history, f"missing E1 col in history: {col}"
