"""NODE_ADSB_HYBRID_V3 architecture — CL-residual longitudinal output (Phase 1.5).

Sibling of :mod:`node_fdm.architectures.adsb_hybrid_v2`. v3 re-parameterises
the longitudinal lift output from the adimensional residual
``fdm_lift_residual_norm`` (``L = (1 + lift_residual_norm) · m·g``) to a CL
residual ``fdm_cl_residual`` (``CL = CL_REF + cl_residual``,
``L = q·S·CL``). State/control/env columns, lateral channel, trajectory
layer, and MassEncoder feature set are byte-identical to v2; only the
longitudinal head output and the PhysicsLayer input contract change.

The identifiability invariant on ``data_ode_long.input_cols`` (same as the
NODE_ADSB_V1 baseline — no mass, no flight features) is preserved.

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
    FLIGHT_FEATURE_COLS_6,
    FLIGHT_FEATURE_SIGNS_6,
    U_COLS,
    X_COLS,
)

__all__ = [
    "NODE_ADSB_HYBRID_V3",
]

# Identifiability invariant (AC4): the longitudinal NN backbone must see the
# exact same inputs as the baseline NODE_ADSB_V1 — no mass, no flight-level
# features. Any loss improvement against the baseline must come from the
# MassEncoder, not from extra inputs leaking into the backbone.
_BASELINE_DATA_ODE_LONG = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_long"
)
_BASELINE_TRAJECTORY = next(layer for layer in NODE_ADSB_V1.layers if layer.name == "trajectory")
_BASELINE_DATA_ODE_LAT = next(
    layer for layer in NODE_ADSB_V1.layers if layer.name == "data_ode_lat"
)

NODE_ADSB_HYBRID_V3 = ArchitectureSpec(
    name="node_adsb_hybrid_v3",
    x_cols=X_COLS,
    u_cols=U_COLS,
    e0_cols=E0_COLS,
    e1_cols=E1_COLS,
    dx_cols=DX_COLS,
    layers=[
        # Trajectory layer is mass-agnostic — reuse the baseline as-is.
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
            # Identical to NODE_ADSB_V1.data_ode_long.input_cols (AC4 invariant).
            input_cols=list(_BASELINE_DATA_ODE_LONG.input_cols),
            # CL-mode contract (Phase 1.5): replace the adimensional lift
            # residual with a CL residual. PhysicsLayer reconstructs Newton
            # forces analytically:
            #   T - D = t_minus_d_norm · m_ref
            #   CL    = CL_REF + cl_residual
            #   L     = q · S · CL
            # CL_REF and S_REF_A320_M2 are constants defined in physics.py
            # (landed by AXM-1737).
            output_cols=["fdm_t_minus_d_norm", "fdm_cl_residual"],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_t_minus_d_norm": "scaled",
                    "fdm_cl_residual": "scaled",
                },
            },
        ),
        # Lateral channel is mass-independent (cf. §4.5) — reuse the baseline.
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
            # CL-mode contract: physics consumes (T-D, cl_residual, m, V, gamma,
            # q, phi_bank) and emits the longitudinal/lateral derivatives.
            # fdm_q_pa is required by the CL branch (L = q·S·CL).
            input_cols=[
                "fdm_t_minus_d_norm",
                "fdm_cl_residual",
                "fdm_mass_kg",
                "era_tas_ms",
                "fdm_gamma_rad",
                "fdm_q_pa",
                "fdm_phi_bank_rad",
            ],
            output_cols=[
                "fdm_d_tas_ms2",
                "fdm_d_gamma_rads",
                "fdm_d_mass_kgs",
                "fdm_d_heading_rads",
                # Reconstructed full lift, exposed for downstream diagnostics.
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
        # Mass should stay constant (dm/dt = 0) but the bound is a safety net
        # for the projected integrator: clamp to A320 TCDS plausible range.
        "fdm_mass_kg": (A320_OEW_KG, A320_MTOW_KG),
    },
    dx_bounds={
        **NODE_ADSB_V1.dx_bounds,
        # Mass is strictly pinned: no fuel-burn modelling at this stage.
        "fdm_d_mass_kgs": (0.0, 0.0),
    },
    derived_output_cols=[
        "fdm_t_minus_d_norm",
        # Phase 1.5: CL residual replaces lift_residual_norm. Sibling
        # decoupling — v3 outputs cl_residual exclusively.
        "fdm_cl_residual",
        # Lateral channel unchanged from baseline.
        "fdm_phi_bank_rad",
    ],
    nn_output_caps={
        "fdm_t_minus_d_norm": 8.0,
        # CL = CL_REF + cl_residual = 0.5 + cl_residual. Cap of 1.0 keeps
        # CL ∈ [-0.5, 1.5], aligning the upper bound on CL_MAX (stall).
        "fdm_cl_residual": 1.0,
        # phi_bank cap = pi/3 (60°) — commercial operating limit, unchanged.
        "fdm_phi_bank_rad": math.pi / 3,
    },
    # Same active-signal filter as the baseline.
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS_6,
    flight_feature_signs=FLIGHT_FEATURE_SIGNS_6,
)

register(NODE_ADSB_HYBRID_V3)
