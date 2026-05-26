"""NODE_ADSB_HYBRID_V19_PSEFFICIENCY_PARALLEL_L025 — Phase 9 : v17 with λ=0.25.

Variant of :mod:`adsb_hybrid_v17_psefficiency_parallel` with the parallel
unbounded thrust head's λ reduced from 0.5 → 0.25. Two reasons motivate
the reduction :

1. **Pre-existing v13 data** shows v13e (λ=0.25) reached val_loss = 0.029495,
   slightly better than v13d (λ=0.5) at 0.029576. The smaller escape is
   already known to be no worse, and might be marginally better.

2. **v17 η-chain pollution** (Phase 8 LESSONS §"What remains open") : v17's
   AC4_thrust mean = 55.7 %, p99 = 159 % — the parallel head fires very
   aggressively. The downstream η-PS chain sees extreme c_t_inst values,
   producing R8 cruise eta_PS min = 0.051 (vs v14's 0.188) and R9 cruise
   mdot_f max = 18.7 kg/s (vs v14's 0.95). Halving λ should mechanically
   reduce the parallel head contribution by 2x, restoring sane η-chain
   tails while keeping val_loss near v9 level.

Implementation : trigger the existing PhysicsLayer dispatch on the
``fdm_t_correction_parallel_l025`` column name (already wired since
v13e — uses ``parallel_lambda = 0.25`` and
``parallel_col = "fdm_t_minus_d_norm_parallel_l025"``). The Phase 4 η
chain is preserved by keeping `fdm_eta_correction` in the head outputs.

6 NN outputs (same count as v17, just different column names for the
parallel pair) :
    fdm_cl_residual,
    fdm_cd_correction,
    fdm_throttle_norm,
    fdm_t_correction_parallel_l025,         ← was _parallel  (now λ=0.25)
    fdm_t_minus_d_norm_parallel_l025,       ← was _parallel  (now λ=0.25)
    fdm_eta_correction.

Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.adsb_hybrid_v14_psefficiency import (
    NODE_ADSB_HYBRID_V14_PSEFFICIENCY,
)
from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register

__all__ = [
    "NODE_ADSB_HYBRID_V19_PSEFFICIENCY_PARALLEL_L025",
]


def _v19_layers() -> list[LayerSpec]:
    """Build the v19 layer list : v14 layers + Strategy W λ=0.25 + η head."""
    layers: list[LayerSpec] = []
    for layer in NODE_ADSB_HYBRID_V14_PSEFFICIENCY.layers:
        if layer.name == "data_ode_long":
            # v17-style head set but with the _l025 column names so
            # PhysicsLayer dispatches to the λ=0.25 branch.
            new_outputs = [
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_throttle_norm",
                "fdm_t_correction_parallel_l025",
                "fdm_t_minus_d_norm_parallel_l025",
                "fdm_eta_correction",
            ]
            new_config = {
                "denormalize_modes": dict.fromkeys(new_outputs, "scaled"),
                "scale_overrides": {
                    "fdm_cd_correction": 1.0,
                    "fdm_throttle_norm": 1.0,
                    "fdm_t_correction_parallel_l025": 1.0,
                    "fdm_t_minus_d_norm_parallel_l025": 1.0,
                    "fdm_eta_correction": 1.0,
                },
                "cap_overrides": {
                    "fdm_cd_correction": 3.0,
                    "fdm_throttle_norm": 5.0,
                    "fdm_t_correction_parallel_l025": 3.0,
                    "fdm_t_minus_d_norm_parallel_l025": 8.0,
                    "fdm_eta_correction": 2.0,
                },
            }
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
            # Same physics columns as v17 but with the _l025 names so the
            # PhysicsLayer dispatch picks parallel_lambda=0.25.
            new_inputs = [
                "fdm_cl_residual",
                "fdm_cd_correction",
                "fdm_throttle_norm",
                "fdm_t_correction_parallel_l025",
                "fdm_t_minus_d_norm_parallel_l025",
                "fdm_eta_correction",
                "fdm_mass_kg",
                "era_tas_ms",
                "fdm_gamma_rad",
                "fdm_phi_bank_rad",
                "era_mach",
                "era_temp_K",
                "fdm_q_pa",
                "raw_alt_m",
            ]
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


NODE_ADSB_HYBRID_V19_PSEFFICIENCY_PARALLEL_L025 = ArchitectureSpec(
    name="node_adsb_hybrid_v19_psefficiency_parallel_l025",
    x_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.x_cols,
    u_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.u_cols,
    e0_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.e0_cols,
    e1_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.e1_cols,
    dx_cols=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.dx_cols,
    layers=_v19_layers(),
    preprocessing_fn=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.preprocessing_fn,
    segment_filter_fn=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.segment_filter_fn,
    x_bounds=dict(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.x_bounds),
    dx_bounds=dict(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.dx_bounds),
    derived_output_cols=list(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.derived_output_cols),
    nn_output_caps={
        **{
            k: v
            for k, v in NODE_ADSB_HYBRID_V14_PSEFFICIENCY.nn_output_caps.items()
            if k != "fdm_t_correction"
        },
        "fdm_t_correction_parallel_l025": 3.0,
        "fdm_t_minus_d_norm_parallel_l025": 8.0,
    },
    nn_output_scale_floor_ratio=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.nn_output_scale_floor_ratio,
    flight_feature_cols=list(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.flight_feature_cols),
    flight_feature_signs=list(NODE_ADSB_HYBRID_V14_PSEFFICIENCY.flight_feature_signs),
    mass_encoder_temperature=NODE_ADSB_HYBRID_V14_PSEFFICIENCY.mass_encoder_temperature,
)

register(NODE_ADSB_HYBRID_V19_PSEFFICIENCY_PARALLEL_L025)
