"""NODE_ADSB_HYBRID_V6_MLP — strict-monotone MLP MassEncoder (Strategy B).

Sibling of :mod:`node_fdm.architectures.adsb_hybrid_v5_tempered`. Same
9-feature set and Newton-mode physics, but the MassEncoder is replaced
with :class:`MassEncoderMLPMonotone` — a 32-hidden-unit MLP whose
positive-softplus weights and ``softplus`` activations make it strictly
monotone in every expected-sign-projected input feature. The pre-sigmoid
temperature stays at ``T = 3.0``.

The MLP keeps the same monotonic prior as the linear encoder but unlocks
non-linear feature interactions — needed when the rank-ordering of
flights by predicted mass under a linear combination disagrees with
truth (cf. Experiment 02 corr ceiling at 0.22 despite a near-perfect
prediction range).

Identifiability invariant (AC4): the longitudinal NN backbone still sees
only the baseline inputs.

Auto-registers at import time.
"""

from __future__ import annotations

import math

from node_fdm.architectures.adsb import NODE_ADSB_V1
from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm_data.schemas.adsb_hybrid import (
    A320_MTOW_KG,
    A320_OEW_KG,
    DX_COLS,
    E0_COLS,
    E1_COLS,
    FLIGHT_FEATURE_COLS_9,
    FLIGHT_FEATURE_SIGNS_9,
    U_COLS,
    X_COLS,
)

__all__ = [
    "NODE_ADSB_HYBRID_V6_MLP",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V6_MLP = ArchitectureSpec(
    name="node_adsb_hybrid_v6_mlp_monotone",
    x_cols=X_COLS,
    u_cols=U_COLS,
    e0_cols=E0_COLS,
    e1_cols=E1_COLS,
    dx_cols=DX_COLS,
    layers=[
        LayerSpec(
            name=_BASELINE_TRAJECTORY.name,
            layer_class=_BASELINE_TRAJECTORY.layer_class,
            input_cols=_BASELINE_TRAJECTORY.input_cols,
            output_cols=_BASELINE_TRAJECTORY.output_cols,
            trainable=_BASELINE_TRAJECTORY.trainable,
            config=_BASELINE_TRAJECTORY.config,
        ),
        LayerSpec(
            name="data_ode_long",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=list(_BASELINE_DATA_ODE_LONG.input_cols),
            output_cols=["fdm_t_minus_d_norm", "fdm_lift_residual_norm"],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_t_minus_d_norm": "scaled",
                    "fdm_lift_residual_norm": "scaled",
                },
            },
        ),
        LayerSpec(
            name=_BASELINE_DATA_ODE_LAT.name,
            layer_class=_BASELINE_DATA_ODE_LAT.layer_class,
            input_cols=_BASELINE_DATA_ODE_LAT.input_cols,
            output_cols=_BASELINE_DATA_ODE_LAT.output_cols,
            trainable=_BASELINE_DATA_ODE_LAT.trainable,
            config=_BASELINE_DATA_ODE_LAT.config,
        ),
        LayerSpec(
            name="physics",
            layer_class="node_fdm.layers.physics.PhysicsLayer",
            input_cols=[
                "fdm_t_minus_d_norm",
                "fdm_lift_residual_norm",
                "fdm_mass_kg",
                "era_tas_ms",
                "fdm_gamma_rad",
                "fdm_phi_bank_rad",
            ],
            output_cols=[
                "fdm_d_tas_ms2",
                "fdm_d_gamma_rads",
                "fdm_d_mass_kgs",
                "fdm_d_heading_rads",
                "fdm_lift_N",
            ],
            trainable=False,
            config={},
        ),
    ],
    preprocessing_fn=NODE_ADSB_V1.preprocessing_fn,
    segment_filter_fn=NODE_ADSB_V1.segment_filter_fn,
    x_bounds={
        **NODE_ADSB_V1.x_bounds,
        "fdm_mass_kg": (A320_OEW_KG, A320_MTOW_KG),
    },
    dx_bounds={
        **NODE_ADSB_V1.dx_bounds,
        "fdm_d_mass_kgs": (0.0, 0.0),
    },
    derived_output_cols=[
        "fdm_t_minus_d_norm",
        "fdm_lift_residual_norm",
        "fdm_phi_bank_rad",
    ],
    nn_output_caps={
        "fdm_t_minus_d_norm": 8.0,
        "fdm_lift_residual_norm": 2.0,
        "fdm_phi_bank_rad": math.pi / 3,
    },
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS_9,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_9,
    mass_encoder_temperature=3.0,
    mass_encoder_class_path="node_fdm.layers.mass_encoder.MassEncoderMLPMonotone",
    mass_encoder_kwargs={"hidden_dim": 32, "temperature": 3.0},
)

register(NODE_ADSB_HYBRID_V6_MLP)
