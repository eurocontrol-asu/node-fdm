"""NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL — Phase 8 Exp 2 : v14 + Strategy W.

The v9 → v14 +25 % val_loss regress survived two falsifications :
Phase 7 (throttle activation, three variants) and Phase 8 Exp 1
(`d_mass = -mdot_f` coupling). Both lifted the val_loss by ≤ 3 % and
did not move the t_correction NN-head off its ±5 % cap (AC4 mean stayed
at ~3.9-4.4 % across all variants).

The leaderboard, however, contains pre-existing Phase 3 variants where
the val_loss **does** drop back to v9 territory :

    v13d (parallel λ=0.5)         best val_loss = 0.029576
    v13e (parallel λ=0.25)        best val_loss = 0.029495
    v13g (TET parallel λ=0.5)     best val_loss = 0.030007

These are Strategy W variants — they keep the v13 bounded NN
t_correction (±5 %) and **add an unbounded `fdm_t_minus_d_norm_parallel`
head** that contributes `λ · t_minus_d_norm · m_ref` to the thrust
total :

    T_total = T_PS · (1 + 0.05·tanh(t_correction))
            + λ · t_minus_d_norm_parallel · m_ref       (unbounded escape)

The unbounded head gives the NN the headroom the bounded ±5 % cap
denies it. v9 had exactly this freedom (only a `T-D` head, no analytical
prior at all) — and that is why v9 val_loss = 0.0298. v14 removed this
escape entirely (no parallel head, only the bounded t_correction +
Phase 4 closure), and val_loss climbed to 0.0372.

**This v17 lifts Strategy W onto the v14 fuel-flow chain.** Six NN
outputs : the v13d 5 outputs (cl_residual, cd_correction, throttle_norm,
t_correction_parallel, t_minus_d_norm_parallel) **plus** v14's
`fdm_eta_correction` for the η-PS closure. PhysicsLayer dispatches on
``fdm_t_correction_parallel`` (Strategy W branch with λ = 0.5) and
then extends into Phase 4 (`fdm_eta_correction` present) for the
η + mdot_f + d_mass = -mdot_f chain. R8 / R9 / AC7 stay valid because
the fuel-flow closure is untouched.

Expected outcome :
    val_loss ≤ 0.032 ✅ → Phase 8 success, Strategy W on Phase 4 closure
                          is the recipe.
    val_loss ∈ (0.032, 0.036] ⚠️ → partial recovery, η-closure has
                          some independent cost.
    val_loss > 0.036 ❌ → the parallel head doesn't transfer ; the
                          regress is in the η-closure itself.

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
    "NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL = ArchitectureSpec(
    name="node_adsb_hybrid_v17_psefficiency_parallel",
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
            # v13d Strategy W (5 heads) + v14 Phase 4 eta_correction (1 head).
            output_cols=[
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_throttle_norm",
                "fdm_t_correction_parallel",
                "fdm_t_minus_d_norm_parallel",
                "fdm_eta_correction",
            ],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_cl_residual": "scaled",
                    "fdm_cd_correction": "scaled",
                    "fdm_throttle_norm": "scaled",
                    "fdm_t_correction_parallel": "scaled",
                    "fdm_t_minus_d_norm_parallel": "scaled",
                    "fdm_eta_correction": "scaled",
                },
                "scale_overrides": {
                    "fdm_cd_correction": 1.0,
                    "fdm_throttle_norm": 1.0,
                    "fdm_t_correction_parallel": 1.0,
                    "fdm_t_minus_d_norm_parallel": 1.0,
                    "fdm_eta_correction": 1.0,
                },
                "cap_overrides": {
                    "fdm_cd_correction": 3.0,
                    "fdm_throttle_norm": 5.0,
                    "fdm_t_correction_parallel": 3.0,
                    "fdm_t_minus_d_norm_parallel": 8.0,
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
                "fdm_t_correction_parallel",
                "fdm_t_minus_d_norm_parallel",
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
        # Phase 4 fuel-burn range — same as v14 (cruise mdot_f ≈ 0.6 kg/s
        # total, climb up to ~1.5 kg/s).
        "fdm_d_mass_kgs": (-2.0, 0.0),
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
        "fdm_t_correction_parallel": 3.0,
        "fdm_t_minus_d_norm_parallel": 8.0,
        "fdm_eta_correction": 2.0,
    },
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS_3_CAUSAL,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_3_CAUSAL,
    mass_encoder_temperature=1.5,
)

register(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL)
