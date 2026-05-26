"""NODE_ADSB_HYBRID_V18_PARALLEL_T_AND_CD — Phase 8 Exp 3 : Strategy W on drag too.

v17 successfully recovered val_loss to v9 level (0.029972) by adding an
unbounded parallel `fdm_t_minus_d_norm_parallel` head on the thrust
channel. The Phase 8 forwardpass diagnostic also confirmed that in
v14 baseline, **both** the t_correction head **and** the cd_correction
head were saturated at their ±5 % caps :

    v14 baseline :  AC4_thrust mean=3.9 %,   p99=4.975 %  ❌
                    AC4_cd    mean=3.962 %, p99=4.975 %  ❌ (also saturated!)

In v17, AC4_cd dropped to 2.078 % — the thrust parallel head absorbed
the demand that cd_correction was previously over-compensating for.
But the cd head still has unused headroom : 5 % is well above its
current 2 % usage, so adding a parallel escape on cd should *not*
unlock new corrective capacity unless there's residual demand the
thrust parallel can't cover.

v18 hypothesis : adding a parallel `fdm_cd_minus_d_norm_parallel`
head on top of v17 either :
- (a) drops val_loss another 1-3 % → drag channel had residual
  bottleneck despite the thrust parallel head ; **the systemic
  pattern "bounded → glued to cap" generalises**.
- (b) leaves val_loss ≈ v17 → drag head was OK as-is in v17, the
  recipe doesn't generalise (this is the "drag wasn't the constraint
  once thrust was freed" reading).

Either result is informative for the paper (positive : design rule
"add parallel escape on every bounded channel" ; negative : "only
thrust needed it because the others were free-rider compensators").

PhysicsLayer (Phase 8 patch) : in the Phase 3 / Phase 4 branch, after
computing the bounded ``d_force = q·S·c_d·(1 + 0.05·tanh(cd_correction))``,
add ``λ · cd_minus·m_ref`` if the column is present. Same λ scalar as
the thrust parallel head (0.5 for ``fdm_t_correction_parallel``-keyed
archs) — symmetric escape on both channels.

7 NN outputs : v17's 6 + ``fdm_cd_minus_d_norm_parallel``.

Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.adsb_hybrid_v17_psefficiency_parallel import (
    NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL,
)
from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register

__all__ = [
    "NODE_ADSB_HYBRID_V18_PARALLEL_T_AND_CD",
]


def _v18_layers() -> list[LayerSpec]:
    """Build the v18 layer list : v17 layers + ``fdm_cd_minus_d_norm_parallel``."""
    layers: list[LayerSpec] = []
    for layer in NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.layers:
        if layer.name == "data_ode_long":
            new_outputs = [*list(layer.output_cols), "fdm_cd_minus_d_norm_parallel"]
            new_config = dict(layer.config)
            for key in ("denormalize_modes", "scale_overrides", "cap_overrides"):
                new_config[key] = dict(new_config[key])
            new_config["denormalize_modes"]["fdm_cd_minus_d_norm_parallel"] = "scaled"
            new_config["scale_overrides"]["fdm_cd_minus_d_norm_parallel"] = 1.0
            new_config["cap_overrides"]["fdm_cd_minus_d_norm_parallel"] = 8.0
            layers.append(
                LayerSpec(
                    name=layer.name,
                    layer_class=layer.layer_class,
                    input_cols=layer.input_cols,
                    output_cols=new_outputs,
                    trainable=layer.trainable,
                    config=new_config,
                )
            )
        elif layer.name == "physics":
            # PhysicsLayer needs the new column as an input so it can
            # apply the parallel cd contribution.
            new_inputs = list(layer.input_cols)
            new_inputs.insert(
                new_inputs.index("fdm_t_minus_d_norm_parallel") + 1,
                "fdm_cd_minus_d_norm_parallel",
            )
            layers.append(
                LayerSpec(
                    name=layer.name,
                    layer_class=layer.layer_class,
                    input_cols=new_inputs,
                    output_cols=layer.output_cols,
                    trainable=layer.trainable,
                    config=layer.config,
                )
            )
        else:
            layers.append(layer)
    return layers


NODE_ADSB_HYBRID_V18_PARALLEL_T_AND_CD = ArchitectureSpec(
    name="node_adsb_hybrid_v18_parallel_t_and_cd",
    x_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.x_cols,
    u_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.u_cols,
    e0_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.e0_cols,
    e1_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.e1_cols,
    dx_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.dx_cols,
    layers=_v18_layers(),
    preprocessing_fn=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.preprocessing_fn,
    segment_filter_fn=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.segment_filter_fn,
    x_bounds=dict(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.x_bounds),
    dx_bounds=dict(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.dx_bounds),
    derived_output_cols=list(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.derived_output_cols),
    nn_output_caps={
        **dict(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.nn_output_caps),
        "fdm_cd_minus_d_norm_parallel": 8.0,
    },
    nn_output_scale_floor_ratio=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.nn_output_scale_floor_ratio,
    flight_feature_cols=list(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.flight_feature_cols),
    flight_feature_signs=list(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.flight_feature_signs),
    mass_encoder_temperature=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.mass_encoder_temperature,
)

register(NODE_ADSB_HYBRID_V18_PARALLEL_T_AND_CD)
