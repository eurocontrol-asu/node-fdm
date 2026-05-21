"""NODE_ADSB_HYBRID_V11_PS_RESIDUAL — Strategy Y (Exp 09).

PINN-residual mass encoder anchored on Poll-Schumann Eq 100. For
cruise-stable segments, the encoder uses ``m_PS_anchor`` (algebraic
inversion of FL_obs through the A320 psi-set) as its baseline ; for
non-cruise segments, it falls back to the v9 linear-tempered encoder
(T=1.5, signs +/-/-, 3 causal features). On top of the anchor, a
small MLP head learns ``Delta_m_NN in [-5 t, +5 t]`` from state at t0
``(alt_norm, dh_dt_norm)``, absorbing the Eq 100 ~+14.5 % bias as a
learned offset and providing local adaptation.

This is the project's PINN-residual idiom (Newton's `lift_residual`,
CL-mode's `cl_residual`) applied to mass : a strong physics baseline
plus a bounded NN correction.

Differences vs v9 / v10 :
    * v9 : pure linear-tempered encoder, no P&S signal.
    * v10 : v9 encoder + Eq 100 MSE aux-loss (Exp 08, plateaued at corr 0.362).
    * v11 : Eq 100 enters the *architecture*, not the loss. The
      residual head learns the bias offset and the within-flight
      micro-corrections locally.

R5 compliant : ``m_PS_anchor`` is computed from ``raw_alt_m`` already
in the trainer's state tensor. No QAR data path.

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
    FLIGHT_FEATURE_COLS_3_CAUSAL,
    FLIGHT_FEATURE_SIGNS_3_CAUSAL,
    U_COLS,
    X_COLS,
)

__all__ = [
    "NODE_ADSB_HYBRID_V11_PS_RESIDUAL",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V11_PS_RESIDUAL = ArchitectureSpec(
    name="node_adsb_hybrid_v11_ps_residual",
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
    flight_feature_cols=FLIGHT_FEATURE_COLS_3_CAUSAL,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_3_CAUSAL,
    mass_encoder_temperature=1.5,
    mass_encoder_class_path="node_fdm.layers.mass_encoder.MassEncoderPSResidual",
    mass_encoder_kwargs={
        "temperature": 1.5,
        "delta_hidden": 16,
        "delta_cap_kg": 5_000.0,
    },
)

register(NODE_ADSB_HYBRID_V11_PS_RESIDUAL)
