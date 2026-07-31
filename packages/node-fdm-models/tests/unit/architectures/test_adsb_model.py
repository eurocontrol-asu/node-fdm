"""Executable smoke test for the official ADS-B architecture."""

from __future__ import annotations

import torch

from node_fdm.models.fdm import FlightDynamicsModel
from node_fdm_models import NODE_ADSB_V1


def test_node_adsb_v1_forward_pass_is_finite() -> None:
    """The published spec composes with the generic runtime end to end."""
    spec = NODE_ADSB_V1
    columns = set(spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols)
    columns.update(name for _, name in spec.dx_cols)
    for layer in spec.layers:
        columns.update(layer.input_cols)
        columns.update(layer.output_cols)
    stats = {
        column: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 1.0, "iqr": 1.0}
        for column in columns
    }
    model = FlightDynamicsModel(spec, stats, model_params=(1, 1, 8))

    x = torch.zeros((2, len(spec.x_cols)), dtype=torch.float32)
    x[:, spec.x_cols.index("raw_alt_m")] = 5_000.0
    x[:, spec.x_cols.index("era_tas_ms")] = 200.0
    u = torch.zeros((2, len(spec.u_cols)), dtype=torch.float32)
    for known in (
        "fdm_gamma_target_known",
        "fdm_tas_target_known",
        "fdm_heading_target_known",
        "fdm_heading_known",
    ):
        u[:, spec.u_cols.index(known)] = 1.0
    e = torch.zeros((2, len(spec.e0_cols)), dtype=torch.float32)
    e[:, spec.e0_cols.index("era_temp_K")] = 250.0

    derivatives = model(x, u, e)

    assert derivatives.shape == (2, len(spec.dx_cols))
    assert torch.isfinite(derivatives).all()
