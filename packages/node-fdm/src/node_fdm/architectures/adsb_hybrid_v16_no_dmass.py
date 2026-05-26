"""NODE_ADSB_HYBRID_V16_NO_DMASS — Phase 8 (1) : freeze the mass channel.

Cloned from ``adsb_hybrid_v14_psefficiency`` with one targeted change :
the PhysicsLayer's Phase 4 branch keeps the analytical η-PS chain and
the fuel-flow diagnostic output `fdm_mdot_f`, but writes
``fdm_d_mass_kgs = 0`` instead of ``-mdot_f``. The integrator therefore
propagates mass at the MassEncoder's t0 estimate, restoring the
Phase 1.5 invariant that v9 used.

This isolates the d_mass-coupling contribution to the v9 → v14 +25 %
val_loss regress. If val_loss recovers to ~0.030 with this single
change, the regress is integrator drift on the mass channel — not a
structural property of v14's analytical-prior architecture.

Origin : recommended by
[`LESSONS_PHASE_7_THROTTLE_ACTIVATION.md`](/Users/gabriel/axm/04-papers/PS_MODEL/LESSONS_PHASE_7_THROTTLE_ACTIVATION.md)
§Recommendations (1) after Phase 7 falsified the throttle-activation
mechanism.

Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.adsb_hybrid_v14_psefficiency import (
    NODE_ADSB_HYBRID_V14_PSEFFICIENCY,
)
from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register

__all__ = [
    "NODE_ADSB_HYBRID_V16_NO_DMASS",
]


def _v16_layers_with_frozen_mass() -> list[LayerSpec]:
    """Build the v16 layer list : v14 layers + physics ``mass_coupling=frozen``."""
    layers: list[LayerSpec] = []
    for layer in NODE_ADSB_HYBRID_V14_PSEFFICIENCY.layers:
        if layer.name == "physics":
            new_config = dict(layer.config)
            new_config["mass_coupling"] = "frozen"
            layers.append(
                LayerSpec(
                    name=layer.name,
                    layer_class=layer.layer_class,
                    input_cols=layer.input_cols,
                    output_cols=layer.output_cols,
                    trainable=layer.trainable,
                    config=new_config,
                )
            )
        else:
            layers.append(layer)
    return layers


NODE_ADSB_HYBRID_V16_NO_DMASS = ArchitectureSpec(
    name="node_adsb_hybrid_v16_no_dmass",
    x_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.x_cols,
    u_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.u_cols,
    e0_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.e0_cols,
    e1_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.e1_cols,
    dx_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.dx_cols,
    layers=_v16_layers_with_frozen_mass(),
    preprocessing_fn=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.preprocessing_fn,
    segment_filter_fn=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.segment_filter_fn,
    x_bounds=dict(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.x_bounds),
    dx_bounds=dict(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.dx_bounds),
    derived_output_cols=list(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.derived_output_cols),
    nn_output_caps=dict(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.nn_output_caps),
    nn_output_scale_floor_ratio=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.nn_output_scale_floor_ratio,
    flight_feature_cols=list(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.flight_feature_cols),
    flight_feature_signs=list(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.flight_feature_signs),
    mass_encoder_temperature=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.mass_encoder_temperature,
)

register(NODE_ADSB_HYBRID_V16_NO_DMASS)
