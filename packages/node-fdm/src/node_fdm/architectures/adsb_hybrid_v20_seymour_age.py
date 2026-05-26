"""NODE_ADSB_HYBRID_V20_SEYMOUR_AGE — Phase 10 : v17 + Seymour age correction.

Builds on :mod:`adsb_hybrid_v17_psefficiency_parallel` (the Phase 8
winner with val_loss ≈ 0.030) by adding the Seymour 2020 logarithmic
aircraft-age correction on the propulsive efficiency η_total :

    eta_total_v20 = eta_total_v17 · (1 - k · ln(raw_age_years + 1))

with k = 0.0128 (canonical Seymour 2020 ; equivalently the original
fuel-flow penalty ``mdot_f · 100 / (100 - 1.28 · ln(age + 1))``). The
log form plateaus the deterioration after the first decade — at age 9
the η-reduction is ~3 %, growing to ~5-6 % at age 30 (vs a linear
extrapolation that would predict +20-30 % at 30, physically implausible).

**Goal** : remove the systematic age-deterioration bias from the
analytical η-PS chain. The canonical k = 0.0128 from Seymour 2020 is
used by default ; on cohorts with QAR fuel-flow ground truth available,
k can be calibrated empirically, but in default operation we trust the
literature value rather than over-fitting to a specific fleet.

### What changes vs v17

- `e0_cols` += `raw_age_years` — triggers the loader's `_attach_age_years`
  join on `raw_icao24` against `data/aircraft_db.csv` (150-aircraft
  registry, fleet median fallback for unmatched ICAOs).
- `physics.input_cols` += `raw_age_years` — PhysicsLayer reads it from
  the per-timestep state dict at forward time.
- `physics.config` = `{"seymour_alpha": 0.0128}` — activates the
  Phase 4 branch logarithmic correction with the Seymour 2020 default.
- All 6 NN outputs unchanged ; t_correction parallel head + Phase 4 η
  chain unchanged.

### Why a fixed canonical k, not a fit

v14/v17 forwardpass diagnostics show the η_correction tanh head is
**already dormant** (AC4_eta = 0.000 % across all architectures). The
analytical η chain is essentially the only model for η_total, so the
age correction must come from physics, not data fitting. Using
the canonical Seymour 2020 coefficient keeps the architecture
operator-agnostic and avoids over-fitting to fleet-specific properties
that wouldn't transfer.

R1 compliance : `data/aircraft_db.csv` is the OpenSky / FAA public
registry, not QAR-derived. Joining it preserves the validation-data
isolation invariant.

Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.adsb_hybrid_v17_psefficiency_parallel import (
    NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL,
)
from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register

__all__ = [
    "NODE_ADSB_HYBRID_V20_SEYMOUR_AGE",
]

# Seymour 2020 canonical coefficient (dimensionless), from the
# fuel-flow penalty formula ``100 / (100 - 1.28 · ln(age + 1))``.
_SEYMOUR_ALPHA_DEFAULT: float = 0.0128


def _v20_layers() -> list[LayerSpec]:
    """Build the v20 layer list : v17 layers + Seymour age correction wiring."""
    layers: list[LayerSpec] = []
    for layer in NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.layers:
        if layer.name == "physics":
            new_inputs = [*layer.input_cols, "raw_age_years"]
            new_config = dict(layer.config)
            new_config["seymour_alpha"] = _SEYMOUR_ALPHA_DEFAULT
            layers.append(
                LayerSpec(
                    name=layer.name,
                    layer_class=layer.layer_class,
                    input_cols=new_inputs,
                    output_cols=layer.output_cols,
                    trainable=layer.trainable,
                    config=new_config,
                )
            )
        else:
            layers.append(layer)
    return layers


NODE_ADSB_HYBRID_V20_SEYMOUR_AGE = ArchitectureSpec(
    name="node_adsb_hybrid_v20_seymour_age",
    x_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.x_cols,
    u_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.u_cols,
    # Append raw_age_years to e0 so the per-timestep vect_dict carries it
    # and PhysicsLayer can read x["raw_age_years"] inside the Phase 4 branch.
    e0_cols=[*NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.e0_cols, "raw_age_years"],
    e1_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.e1_cols,
    dx_cols=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.dx_cols,
    layers=_v20_layers(),
    preprocessing_fn=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.preprocessing_fn,
    segment_filter_fn=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.segment_filter_fn,
    x_bounds=dict(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.x_bounds),
    dx_bounds=dict(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.dx_bounds),
    derived_output_cols=list(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.derived_output_cols),
    nn_output_caps=dict(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.nn_output_caps),
    nn_output_scale_floor_ratio=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.nn_output_scale_floor_ratio,
    flight_feature_cols=list(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.flight_feature_cols),
    flight_feature_signs=list(NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.flight_feature_signs),
    mass_encoder_temperature=NODE_ADSB_HYBRID_V17_PSEFFICIENCY_PARALLEL.mass_encoder_temperature,
)

register(NODE_ADSB_HYBRID_V20_SEYMOUR_AGE)
