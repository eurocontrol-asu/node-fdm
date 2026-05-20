"""NODE_ADSB_HYBRID_V7_LEAN — lean 2-feature MassEncoder (Experiment 04).

Sibling of :mod:`node_fdm.architectures.adsb_hybrid_v5_tempered`. Strips
the 9-feature set down to the **2 features the QAR LOOCV oracle picks
first** in greedy forward selection:

* ``cruise_alt_max_flight`` (sign -1, heavier → lower cruise altitude)
* ``dist_total_flight`` (sign +1, heavier → longer planned route)

The oracle reports LOOCV corr = 0.608 on those two features alone — a
ceiling well above the project's 0.5 success threshold. Trained
encoders v4-v6 with 9 features only reached corr 0.14-0.25 because
the extra features added noise the encoder could not reliably exploit
on the small validation set. Stripping to 2 features tests whether
the encoder can recover the oracle's signal under indirect trajectory
supervision.

Newton-mode physics. ``MassEncoderLinearTempered`` with ``T = 3.0`` —
same proven anti-saturation choice as v5.

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
    FLIGHT_FEATURE_COLS_2_LEAN,
    FLIGHT_FEATURE_SIGNS_2_LEAN,
    U_COLS,
    X_COLS,
)

__all__ = [
    "NODE_ADSB_HYBRID_V7_LEAN",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V7_LEAN = ArchitectureSpec(
    name="node_adsb_hybrid_v7_lean_2features",
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
    flight_feature_cols=FLIGHT_FEATURE_COLS_2_LEAN,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_2_LEAN,
    mass_encoder_temperature=3.0,
)

register(NODE_ADSB_HYBRID_V7_LEAN)
