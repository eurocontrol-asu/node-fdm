"""NODE_ADSB_HYBRID_V1 architecture — Newton-force outputs + MassEncoder.

Hybrid variant of :mod:`node_fdm.architectures.adsb`. The longitudinal NN
backbone now emits Newton-force targets ``fdm_t_minus_d_N`` (thrust minus
drag) and ``fdm_lift_N`` instead of the kinematic acceleration
``fdm_a_spec_ms2`` / load factor ``fdm_n_z_residual``. The state vector
gains a fifth dimension ``fdm_mass_kg`` whose derivative is pinned to zero
(mass is held constant inside the horizon; a MassEncoder estimates it from
per-flight features and feeds the projected integrator).

The NN-backbone inputs are kept **identical to** ``NODE_ADSB_V1`` to
preserve the identifiability invariant: any loss improvement at this stage
must come from the MassEncoder, not from new features leaking into the
backbone (cf. PHASE_1_MASS_ENCODER.md §4.4, §5, §7.1).

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
    FLIGHT_FEATURE_COLS,
    U_COLS,
    X_COLS,
)

__all__ = [
    "NODE_ADSB_HYBRID_V1",
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

NODE_ADSB_HYBRID_V1 = ArchitectureSpec(
    name="node_adsb_hybrid_v1",
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
            output_cols=["fdm_t_minus_d_N", "fdm_lift_N"],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_t_minus_d_N": "scaled",
                    "fdm_lift_N": "scaled",
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
            # Newton-form: physics consumes (T-D, L, m, V, gamma, phi_bank)
            # and emits the longitudinal/lateral derivatives directly.
            # n_z is no longer an NN output — it can be recomputed from L/(m·g)
            # downstream if needed.
            input_cols=[
                "fdm_t_minus_d_N",
                "fdm_lift_N",
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
        # NN-output targets — stats derived analytically from observable
        # derivatives via the inverse PhysicsLayer (Newton form).
        "fdm_t_minus_d_N",
        "fdm_lift_N",
        # Lateral channel unchanged from baseline.
        "fdm_phi_bank_rad",
    ],
    nn_output_caps={
        # Newton-force hard caps — leave room above the data-driven p99.9
        # scale so ``cap > scale`` keeps the tanh gradient alive over the
        # full operational envelope (PHASE_1_MASS_ENCODER.md §7.1).
        "fdm_t_minus_d_N": 5.0e5,
        "fdm_lift_N": 1.5e6,
        # phi_bank cap = pi/3 (60°) — commercial operating limit, unchanged.
        "fdm_phi_bank_rad": math.pi / 3,
    },
    # Same active-signal filter as the baseline: Newton-force scales are
    # diluted by long cruise stretches just like a_spec was.
    nn_output_scale_floor_ratio=0.01,
    flight_feature_cols=FLIGHT_FEATURE_COLS,
)

register(NODE_ADSB_HYBRID_V1)
