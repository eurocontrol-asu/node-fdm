"""NODE_ADSB_HYBRID_V12_PSDRAG — Phase 2 PSDragLayer activation.

Cloned from `node_adsb_hybrid_v9_causal_3features` (which remains the
operational baseline MassEncoder). Adds the Poll-Schumann analytical
drag polar as an explicit physics term, with a bounded `+/-5 %` NN
correction (`fdm_cd_correction`). The longitudinal head's
`fdm_t_minus_d_norm` output is re-interpreted as the thrust component
since drag is now supplied analytically — the NN is forced to isolate
thrust from drag.

Differences vs v9 :
    * `data_ode_long.output_cols` adds `fdm_cd_correction` (new third
      output alongside `fdm_t_minus_d_norm` + `fdm_cl_residual`).
    * `physics.input_cols` adds `fdm_cd_correction`, `era_mach`,
      `era_temp_K`, `fdm_q_pa` so the PhysicsLayer can call
      `PSDragLayer(c_l, M, T, q, V)` at runtime.
    * `mass_encoder_temperature` is inherited from v9 (1.5) and the
      MassEncoder class is the v9 linear-tempered one (3 causal features).

Key acceptance criteria (ticket §5) :
    * AC2  : `C_D_PS` within 2 % of poll_schumann_lib `full_polar`
             across A320 cruise envelope. **Verified at the
             PSDragLayer unit test** — max err 0.73 % at C_L=0.7.
    * AC3  : val_loss within 5 % of Phase 1.5 baseline (v9 = 0.0298).
    * AC4  : `mean(|ΔC_D_NN|) < 3 %` of the analytical prior.
    * AC5  : disabling NN drag (`ΔC_D_NN = 0`) does not collapse val_loss.

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
    "NODE_ADSB_HYBRID_V12_PSDRAG",
]

_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V12_PSDRAG = ArchitectureSpec(
    name="node_adsb_hybrid_v12_psdrag",
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
                "fdm_t_minus_d_norm",
                "fdm_cl_residual",
                "fdm_cd_correction",
            ],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_t_minus_d_norm": "scaled",
                    "fdm_cl_residual": "scaled",
                    "fdm_cd_correction": "scaled",
                },
                # ``fdm_cd_correction`` is a synthetic NN output with no
                # observable target. Provide stats via scale_overrides /
                # cap_overrides so the dataset doesn't need to invent
                # one through DERIVED_FEATURES.
                "scale_overrides": {
                    "fdm_cd_correction": 1.0,
                },
                "cap_overrides": {
                    "fdm_cd_correction": 3.0,
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
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_mass_kg",
                "era_tas_ms",
                "fdm_gamma_rad",
                "fdm_phi_bank_rad",
                # Phase 2 — PSDragLayer needs Mach, temperature, q at runtime.
                "era_mach",
                "era_temp_K",
                "fdm_q_pa",
            ],
            output_cols=[
                "fdm_d_tas_ms2",
                "fdm_d_gamma_rads",
                "fdm_d_mass_kgs",
                "fdm_d_heading_rads",
                "fdm_lift_N",
                "fdm_drag_N",
                "fdm_c_d_ps",
                "fdm_c_d_total",
                "fdm_thrust_N",
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
        "fdm_cl_residual",
        "fdm_phi_bank_rad",
        # ``fdm_cd_correction`` is intentionally omitted -- it has no
        # observable target ; its stats come from scale_overrides above.
    ],
    nn_output_caps={
        "fdm_t_minus_d_norm": 8.0,
        "fdm_cl_residual": 0.5,
        "fdm_phi_bank_rad": math.pi / 3,
        # cap=3.0 keeps tanh gradient alive over the +/-5% drag-correction
        # envelope (saturates at tanh(3)=0.995 -> ~4.97% drag tweak).
        "fdm_cd_correction": 3.0,
    },
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS_3_CAUSAL,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_3_CAUSAL,
    mass_encoder_temperature=1.5,
)

register(NODE_ADSB_HYBRID_V12_PSDRAG)
