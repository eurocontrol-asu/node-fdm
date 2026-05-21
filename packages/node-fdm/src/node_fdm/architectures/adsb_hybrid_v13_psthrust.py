"""NODE_ADSB_HYBRID_V13_PSTHRUST — Phase 3 PSThrustLayer activation.

Cloned from `node_adsb_hybrid_v12_psdrag` (Phase 2). Drops the legacy
`fdm_t_minus_d_norm` longitudinal head and adds two new outputs :

    * `fdm_throttle_norm`  — pre-sigmoid latent throttle, mapped to
                              chi in [0.1, 1.0] inside PhysicsLayer.
    * `fdm_t_correction`   — pre-tanh thrust correction, bounded
                              ±5 % via ``0.05 · tanh(...)``.

The PhysicsLayer Phase 3 branch (triggered by presence of
`fdm_t_correction`) computes :

    T_PS = ps_thrust(chi, M, alt, T_inf, q_inf)
    T    = T_PS · (1 + 0.05 · tanh(t_correction))
    D    = q·S·C_D_PS · (1 + 0.05 · tanh(cd_correction))     [Phase 2]
    L    = q·S·(CL_baseline(q, m) + cl_residual)              [Phase 1.5]
    d_TAS   = (T - D) / m - g·sin gamma
    d_gamma = (L/m - g·cos gamma) / V

Key acceptance criteria (PHASE_3_THRUST_PS_TICKET §5) :
    * AC2  : T_PS layer parity vs ps_engine reference < 5 % on
             (FL x M x chi) grid. **Already verified at Exp 11**, 0.000 %.
    * AC3  : val_loss within 5 % of Phase 2 baseline (v12 = 0.0304),
             i.e. ≤ 0.0319.
    * AC4  : mean(|ΔT_NN| / T_PS) < 3 % in cruise.
    * AC5  : disabling NN (`t_correction = 0`) does not collapse val_loss.
    * AC6  : chi (throttle latent) in [0.1, 1.0] and phase-coherent.
    * AC7  : |T_NN_v13|_cruise / |t_minus_d_v12·m_ref|_cruise ∈ [0.5, 2.0].

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
    "NODE_ADSB_HYBRID_V13_PSTHRUST",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V13_PSTHRUST = ArchitectureSpec(
    name="node_adsb_hybrid_v13_psthrust",
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
            # Phase 3 longitudinal contract:
            #   * cl_residual         : Phase 1.5 lift residual (unchanged)
            #   * cd_correction       : Phase 2 drag NN tweak (unchanged, ±5 %)
            #   * throttle_norm       : Phase 3 latent throttle (sigmoid → [0.1, 1.0])
            #   * t_correction        : Phase 3 thrust NN tweak (tanh → ±5 %)
            # `fdm_t_minus_d_norm` is *retiré* — thrust is analytical via T_PS.
            output_cols=[
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_throttle_norm",
                "fdm_t_correction",
            ],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_cl_residual": "scaled",
                    "fdm_cd_correction": "scaled",
                    "fdm_throttle_norm": "scaled",
                    "fdm_t_correction": "scaled",
                },
                # All three "synthetic" outputs (no observable target) need
                # scale + cap overrides because they don't show up in
                # DERIVED_FEATURES.
                #
                # throttle_norm : passes through sigmoid in PhysicsLayer to
                # [0.1, 1.0]. cap=5.0 keeps the gradient alive across the
                # full physical envelope (sigmoid(5)=0.993, sigmoid(-5)=0.007).
                #
                # t_correction : same idiom as cd_correction. cap=3.0 saturates
                # tanh at ±0.995 → 4.97 % absolute thrust tweak (within ±5 %).
                "scale_overrides": {
                    "fdm_cd_correction": 1.0,
                    "fdm_throttle_norm": 1.0,
                    "fdm_t_correction": 1.0,
                },
                "cap_overrides": {
                    "fdm_cd_correction": 3.0,
                    "fdm_throttle_norm": 5.0,
                    "fdm_t_correction": 3.0,
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
                "fdm_mass_kg",
                "era_tas_ms",
                "fdm_gamma_rad",
                "fdm_phi_bank_rad",
                # Phase 2 — PSDragLayer inputs.
                "era_mach",
                "era_temp_K",
                "fdm_q_pa",
                # Phase 3 — PSThrustLayer also takes altitude (accepted for
                # API symmetry ; the layer doesn't use it directly because
                # q_pa already encodes ambient density).
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
        # ``cd_correction``, ``throttle_norm``, ``t_correction`` are
        # *intentionally* omitted -- they have no observable target ; their
        # stats come from the layer's scale_overrides above.
    ],
    nn_output_caps={
        "fdm_cl_residual": 0.5,
        "fdm_phi_bank_rad": math.pi / 3,
        "fdm_cd_correction": 3.0,
        "fdm_throttle_norm": 5.0,
        "fdm_t_correction": 3.0,
    },
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS_3_CAUSAL,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_3_CAUSAL,
    mass_encoder_temperature=1.5,
)

register(NODE_ADSB_HYBRID_V13_PSTHRUST)
