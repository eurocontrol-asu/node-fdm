"""NODE_ADSB_HYBRID_V15_THROTTLE_CLAMP — Phase 7 Strategy A : linear-clamp throttle.

Cloned from ``adsb_hybrid_v14_psefficiency`` with one targeted change :
the PhysicsLayer's Phase 3 throttle activation is switched from
``0.1 + 0.9 * sigmoid(z)`` (v14) to a **linear clamp** :

    χ = (0.1 + 0.9 · (z + 5) / 10).clamp(0.1, 1.0)

with dχ/dz = 0.09 uniform inside z ∈ [-5, +5]. The motivation is to
falsify-or-confirm the hypothesis (LESSONS_PHASE_6_C_T_DO_FALSIFIED.md
§5) that the sigmoid's x13 gradient compression at the χ extrema (idle
χ ≈ 0.11, full thrust χ ≈ 0.99) is responsible for the +25 % val_loss
regress v9 (0.0298) → v14 (0.0372).

Everything else is identical to v14 : same lift / drag / efficiency
closures, same dx_bounds, same NN output caps, same mass encoder
temperature.

Key acceptance criteria (PHASE_7_THROTTLE_ACTIVATION_TICKET §8) :
    * AC5  : val_loss ≤ 0.034 on best epoch (gain ≥ 8 % vs v14).
    * AC6  : AC4_thrust mean ``|ΔT_NN|/T_PS < 3 %`` — NN dé-saturé.
    * AC7  : R6 within-flight corr ≥ +0.70 — mass encoder preserved.
    * AC8  : AC5 NN-off (all 4 heads zeroed) < +50 % degradation.

Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.adsb_hybrid_v14_psefficiency import (
    NODE_ADSB_HYBRID_V14_PSEFFICIENCY,
)
from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register

__all__ = [
    "NODE_ADSB_HYBRID_V15_THROTTLE_CLAMP",
]


def _v15_layers_with_clamp() -> list[LayerSpec]:
    """Build the v15 layer list : v14 layers + physics ``throttle_activation``."""
    layers: list[LayerSpec] = []
    for layer in NODE_ADSB_HYBRID_V14_PSEFFICIENCY.layers:
        if layer.name == "physics":
            new_config = dict(layer.config)
            new_config["throttle_activation"] = "clamp"
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


NODE_ADSB_HYBRID_V15_THROTTLE_CLAMP = ArchitectureSpec(
    name="node_adsb_hybrid_v15_throttle_clamp",
    x_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.x_cols,
    u_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.u_cols,
    e0_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.e0_cols,
    e1_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.e1_cols,
    dx_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.dx_cols,
    layers=_v15_layers_with_clamp(),
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

register(NODE_ADSB_HYBRID_V15_THROTTLE_CLAMP)
