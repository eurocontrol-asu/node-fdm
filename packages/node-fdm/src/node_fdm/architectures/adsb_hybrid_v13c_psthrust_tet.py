"""NODE_ADSB_HYBRID_V13C_PSTHRUST_TET — Phase 3 with TET-nonlinear T_PS.

Variant of `node_adsb_hybrid_v13_psthrust` (Phase 3 baseline). The
analytical thrust prior is upgraded from the linear-throttle Eq 28
model to the full P&S Part 3 Appendix E E3-E5 chain :

    TET(chi) = TET_idle + chi · (TET_MCC - TET_idle)
    T_R      = (1/T_REC) · (TET/T_inf) · [1 - 0.53·(M - M_EC)²] / (1 + 0.2·M²)
    h_4      = 1 + 2.5·(T_R - 1)
    C_T_inst = chi · h_4 · (C_T)_etaB(M)

At typical cruise (FL370, M=0.78, T_inf=219K, full throttle), h_4 ≈ 1.28
— extending the analytical max thrust by ~28 % over the linear model.

Per `experiments/15_qar_t_observable.md` : 17 % of cruise samples need
T_obs > T_PS_linear_max (p90 deficit 19.9 %). The TET-upgrade should
cover this regime *within* the analytical model, freeing the bounded
±5 % NN correction to behave like Phase 2's PSDragLayer correction
(safety margin, not binding constraint).

Triggers Phase 3 branch via `fdm_t_correction_tet` (new column name —
PhysicsLayer dispatches on it to use `PSThrustLayer(mode='tet_e3e5')`).
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
    "NODE_ADSB_HYBRID_V13C_PSTHRUST_TET",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V13C_PSTHRUST_TET = ArchitectureSpec(
    name="node_adsb_hybrid_v13c_psthrust_tet",
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
            output_cols=[
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_throttle_norm",
                "fdm_t_correction_tet",
            ],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_cl_residual": "scaled",
                    "fdm_cd_correction": "scaled",
                    "fdm_throttle_norm": "scaled",
                    "fdm_t_correction_tet": "scaled",
                },
                "scale_overrides": {
                    "fdm_cd_correction": 1.0,
                    "fdm_throttle_norm": 1.0,
                    "fdm_t_correction_tet": 1.0,
                },
                "cap_overrides": {
                    "fdm_cd_correction": 3.0,
                    "fdm_throttle_norm": 5.0,
                    "fdm_t_correction_tet": 3.0,
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
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_throttle_norm",
                "fdm_t_correction_tet",
                "fdm_mass_kg",
                "era_tas_ms",
                "fdm_gamma_rad",
                "fdm_phi_bank_rad",
                "era_mach",
                "era_temp_K",
                "fdm_q_pa",
                "raw_alt_m",
            ],
            output_cols=[
                "fdm_d_tas_ms2",
                "fdm_d_gamma_rads",
                "fdm_d_mass_kgs",
                "fdm_d_heading_rads",
                "fdm_lift_N",
                "fdm_drag_N",
                "fdm_thrust_N",
                "fdm_T_PS",
                "fdm_T_total",
                "fdm_throttle",
                "fdm_c_d_ps",
                "fdm_c_d_total",
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
        "fdm_cl_residual",
        "fdm_phi_bank_rad",
    ],
    nn_output_caps={
        "fdm_cl_residual": 0.5,
        "fdm_phi_bank_rad": math.pi / 3,
        "fdm_cd_correction": 3.0,
        "fdm_throttle_norm": 5.0,
        "fdm_t_correction_tet": 3.0,
    },
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS_3_CAUSAL,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_3_CAUSAL,
    mass_encoder_temperature=1.5,
)

register(NODE_ADSB_HYBRID_V13C_PSTHRUST_TET)
