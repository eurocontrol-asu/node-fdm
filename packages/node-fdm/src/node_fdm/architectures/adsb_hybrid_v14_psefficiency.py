"""NODE_ADSB_HYBRID_V14_PSEFFICIENCY — Phase 4 fuel-flow analytical closure.

Cloned from `node_adsb_hybrid_v13_psthrust` (Phase 3). Adds a fifth NN
output :

    * `fdm_eta_correction` — pre-tanh efficiency correction, bounded
                              ±3 % via ``0.03 * tanh(...)``. Tighter than
                              the ±5 % used in Phase 2 / 3 because eta is
                              more tightly physically constrained.

The PhysicsLayer Phase 4 branch (triggered by presence of
`fdm_eta_correction`) extends the Phase 3 forward chain with :

    C_T_inst = T_total / (q·S_ref)                            [Eq 17 inverse]
    eta_PS   = ps_efficiency(C_T_inst, M)                      [Eqs 24-29]
    eta      = eta_PS * (1 + 0.03 * tanh(eta_correction))
    mdot_f   = T_total * V / (eta_safe * LCV_KEROSENE)         [Eq 19]
    d_mass   = -mdot_f                                          [Phase 1.5 invariant broken]

Key acceptance criteria (PHASE_4_FUEL_FLOW_TICKET §8) :
    * AC2  : PSEfficiencyLayer parity vs ps_engine reference < 2 % on
             (C_T x M) grid. **Verified at Exp 1**, max err 0.0034 %.
    * AC3  : val_loss ≤ v13 + 5 %, i.e. ≤ 0.0389.
    * AC4  : mean(|Δη_NN| / η_o) < 2 % in cruise.
    * AC5  : disabling NN (4 heads = 0) does not collapse val_loss.
    * AC6  : mdot_f cruise in [1.8, 3.2] t/h total A320 (R9 sanity range).
    * AC7  : corr(mdot_f_predicted, FUEL__FF_QAR) ≥ +0.50 on 526 vols.

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
    "NODE_ADSB_HYBRID_V14_PSEFFICIENCY",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V14_PSEFFICIENCY = ArchitectureSpec(
    name="node_adsb_hybrid_v14_psefficiency",
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
            # Phase 4 longitudinal contract:
            #   * cl_residual         : Phase 1.5 lift residual (unchanged)
            #   * cd_correction       : Phase 2 drag NN tweak (unchanged, ±5 %)
            #   * throttle_norm       : Phase 3 latent throttle (sigmoid → [0.1, 1.0])
            #   * t_correction        : Phase 3 thrust NN tweak (tanh → ±5 %)
            #   * eta_correction      : Phase 4 efficiency NN tweak (tanh → ±3 %)
            output_cols=[
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_throttle_norm",
                "fdm_t_correction",
                "fdm_eta_correction",
            ],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_cl_residual": "scaled",
                    "fdm_cd_correction": "scaled",
                    "fdm_throttle_norm": "scaled",
                    "fdm_t_correction": "scaled",
                    "fdm_eta_correction": "scaled",
                },
                # eta_correction passes through tanh in PhysicsLayer to ±cap.
                # cap=2.0 saturates tanh at ±0.964 → 0.03 * 0.964 = 2.89 %
                # absolute efficiency tweak (within the ±3 % envelope).
                # Slightly tighter than t_correction's cap=3.0 because the
                # eta correction has a narrower physical bound.
                "scale_overrides": {
                    "fdm_cd_correction": 1.0,
                    "fdm_throttle_norm": 1.0,
                    "fdm_t_correction": 1.0,
                    "fdm_eta_correction": 1.0,
                },
                "cap_overrides": {
                    "fdm_cd_correction": 3.0,
                    "fdm_throttle_norm": 5.0,
                    "fdm_t_correction": 3.0,
                    "fdm_eta_correction": 2.0,
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
                "fdm_t_correction",
                "fdm_eta_correction",
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
                # Phase 4 diagnostic outputs.
                "fdm_eta_PS",
                "fdm_eta_total",
                "fdm_mdot_f",
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
        # Phase 4 : break the Phase 1.5 invariant `d_mass_kgs = 0`. Cruise
        # fuel-burn for A320 is ~1 kg/s total ; -2.0 leaves headroom for
        # climb (higher mdot_f) without letting the integrator drift into
        # non-physical mass loss.
        "fdm_d_mass_kgs": (-2.0, 0.0),
    },
    derived_output_cols=[
        "fdm_cl_residual",
        "fdm_phi_bank_rad",
        # cd_correction, throttle_norm, t_correction, eta_correction are
        # intentionally omitted -- they have no observable target ; their
        # stats come from the layer's scale_overrides above.
    ],
    nn_output_caps={
        "fdm_cl_residual": 0.5,
        "fdm_phi_bank_rad": math.pi / 3,
        "fdm_cd_correction": 3.0,
        "fdm_throttle_norm": 5.0,
        "fdm_t_correction": 3.0,
        "fdm_eta_correction": 2.0,
    },
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS_3_CAUSAL,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_3_CAUSAL,
    mass_encoder_temperature=1.5,
)

register(NODE_ADSB_HYBRID_V14_PSEFFICIENCY)
