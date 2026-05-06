"""ADS-B v1 architecture specification.

Simplified two-layer architecture (trajectory + data_ode + physics) for
ADS-B data.  Compared to legacy OpenSky architectures:

* Smaller state vector (no cumulative distance).
* Robust altitude control (``fdm_alt_target_m``, never NaN) kept in
  ``U_COLS`` for the ``TrajectoryLayer``; the ``StructuredLayer`` receives
  only ``U_ODE_COLS`` (``U_COLS`` minus ``fdm_alt_target_m``).
* Leaner environment (no airport distances).
* Fewer derivatives (no ground-speed derivative).

Phase 2B adds the lateral channel: heading state, heading target,
phi_bank NN command, ``d_heading = (g/V)·tan(phi_bank)`` physics, and
the wind-triangle drift / track / 2D ground-speed recomputation in the
TrajectoryLayer.

Auto-registers at import time.
"""

from __future__ import annotations

import math

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm_data.schemas.adsb import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

__all__ = [
    "NODE_ADSB_V1",
]

NODE_ADSB_V1 = ArchitectureSpec(
    name="node_adsb_v1",
    x_cols=X_COLS,
    u_cols=U_COLS,
    e0_cols=E0_COLS,
    e1_cols=E1_COLS,
    dx_cols=DX_COLS,
    layers=[
        LayerSpec(
            name="trajectory",
            layer_class="node_fdm.layers.trajectory.TrajectoryLayer",
            input_cols=X_COLS + U_COLS + E0_COLS,
            output_cols=[
                *E1_COLS,
                "fdm_gamma_target_known",
                "fdm_heading_target_known",
                "fdm_heading_known",
            ],
            trainable=False,
            config={
                "col_map": {
                    "tas": "era_tas_ms",
                    "gamma": "fdm_gamma_rad",
                    "alt": "raw_alt_m",
                    "wind": "fdm_long_wind_ms",
                    "temp": "era_temp_K",
                    "alt_sel": "fdm_alt_target_m",
                    "vz": "fdm_d_alt_ms",
                    "gs": "raw_gs_ms",
                    "mach": "era_mach",
                    "cas": "fdm_cas_ms",
                    "alt_diff": "fdm_alt_diff_m",
                    "tas_sel": "fdm_tas_target_ms",
                    "tas_known": "fdm_tas_target_known",
                    "tas_diff": "fdm_tas_diff_ms",
                    "gamma_sel": "fdm_gamma_target_rad",
                    "gamma_known": "fdm_gamma_target_known",
                    "gamma_diff": "fdm_gamma_diff_rad",
                    "g_sin_gamma": "fdm_g_sin_gamma_ms2",
                    "cos_gamma": "fdm_cos_gamma",
                    "q": "fdm_q_pa",
                    "g_over_v": "fdm_g_over_v",
                    # Lateral channel (Phase 2B)
                    "u_wind": "era_u_wind_ms",
                    "v_wind": "era_v_wind_ms",
                    "heading": "fdm_heading_rad",
                    "heading_target": "fdm_heading_target_rad",
                    "heading_target_known": "fdm_heading_target_known",
                    "heading_known": "fdm_heading_known",
                    "lat_wind": "fdm_lat_wind_ms",
                    "drift": "fdm_drift_rad",
                    "track": "fdm_track_rad",
                    "heading_diff": "fdm_heading_diff_rad",
                },
            },
        ),
        LayerSpec(
            name="data_ode_long",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            # Longitudinal backbone: long-pure inputs only (no heading state,
            # no heading target, no wind-triangle lat features).
            input_cols=[
                # X (long)
                "raw_alt_m",
                "fdm_gamma_rad",
                "era_tas_ms",
                # U_ODE = []
                # E0 (long)
                "fdm_long_wind_ms",
                "era_temp_K",
                # E1 (long)
                "fdm_d_alt_ms",
                "era_mach",
                "raw_gs_ms",
                "fdm_cas_ms",
                "fdm_alt_diff_m",
                "fdm_tas_diff_ms",
                "fdm_gamma_diff_rad",
                "fdm_g_sin_gamma_ms2",
                "fdm_cos_gamma",
                "fdm_q_pa",
                "fdm_g_over_v",
                # flags long
                "fdm_gamma_target_known",
                "fdm_tas_target_known",
            ],
            output_cols=["fdm_a_spec_ms2", "fdm_n_z_residual"],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_a_spec_ms2": "scaled",
                    "fdm_n_z_residual": "scaled",
                },
                # Scale/cap are now sourced from compute_stats: each col is
                # listed in ``derived_output_cols`` below, DERIVED_FEATURES
                # inverts the PhysicsLayer to produce per-sample values, and
                # the resulting p999 feeds OutputDenormalizer (cap = p999,
                # so denorm = p999 * tanh(x / p999)).
            },
        ),
        LayerSpec(
            name="data_ode_lat",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            # Lateral backbone: lat-pure cols + minimal physical coupling
            # (Option C — era_tas_ms for the g/V gain, raw_alt_m for the
            # flight regime, fdm_gamma_rad for the 1/cos(gamma) factor).
            input_cols=[
                # State / control / E1 lat-pures.
                # fdm_heading_target_rad is intentionally NOT included: it
                # carries NaN where target_known=False (sentinel from
                # derive.py), and U columns are exempted from the loader's
                # NaN-window filter on purpose (gamma/tas/heading targets
                # are expected to be NaN). The signal is available via the
                # already-sanitized fdm_heading_diff_rad (computed by
                # TrajectoryLayer with nan_to_num). The known-flag
                # fdm_heading_target_known lets the head learn when to
                # ignore the diff.
                "fdm_heading_rad",
                "fdm_heading_target_known",
                "fdm_heading_known",
                "era_u_wind_ms",
                "era_v_wind_ms",
                "fdm_lat_wind_ms",
                "fdm_drift_rad",
                "fdm_track_rad",
                "fdm_heading_diff_rad",
                # Couplage physique 1er ordre (option C)
                "era_tas_ms",
                "raw_alt_m",
                "fdm_gamma_rad",
            ],
            output_cols=["fdm_phi_bank_rad"],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_phi_bank_rad": "scaled",
                },
                # phi_bank cap = 1.0 rad is a physics/conditioning decision,
                # not a data percentile (tan(1.0)=1.557 keeps ``g/V·tan``
                # well-conditioned; dx clamp ±0.1 rad/s bounds the rate).
                # Carried at the spec level via ``nn_output_caps`` so it
                # overrides the data-driven p999 fallback while still
                # appearing in stats for diagnostics.
            },
        ),
        LayerSpec(
            name="physics",
            layer_class="node_fdm.layers.physics.PhysicsLayer",
            input_cols=[
                "fdm_a_spec_ms2",
                "fdm_n_z_residual",
                "era_tas_ms",
                "fdm_gamma_rad",
                "fdm_phi_bank_rad",
            ],
            output_cols=[
                "fdm_d_tas_ms2",
                "fdm_d_gamma_rads",
                "fdm_n_z",
                "fdm_d_heading_rads",
            ],
            trainable=False,
            config={},
        ),
    ],
    preprocessing_fn="node_fdm_data.preprocessing.opensky.flight_processing",
    segment_filter_fn=None,
    x_bounds={
        "raw_alt_m": (-500.0, 20000.0),
        "fdm_gamma_rad": (-0.3, 0.3),
        "era_tas_ms": (0.0, 350.0),
        # Note (Decision 4): fdm_heading_rad is intentionally NOT bounded.
        # The projected integrator clamps state to bounds, which would clip
        # a free-running 6.5-rad heading down to 6.28 instead of wrapping
        # to 0.22.  Letting the state evolve freely keeps the trajectory
        # smooth; the wrap-aware loss in the trainer handles the comparison
        # and the cos/sin used by drift/track are naturally 2π-periodic.
    },
    dx_bounds={
        "fdm_d_alt_ms": (-50.0, 50.0),
        "fdm_d_gamma_rads": (-0.03, 0.03),
        "fdm_d_tas_ms2": (-12.5, 12.5),
        # Rate-2 turn (6 deg/s = 0.1 rad/s) is the typical commercial bound.
        "fdm_d_heading_rads": (-0.1, 0.1),
    },
    derived_output_cols=[
        # NN-output targets — stats derived from observable derivatives via
        # the inverse PhysicsLayer (see node_fdm.dataset.DERIVED_FEATURES).
        "fdm_a_spec_ms2",
        "fdm_n_z_residual",
        "fdm_phi_bank_rad",
    ],
    nn_output_caps={
        # Hard regulatory / physical bounds — not data-driven. Each cap
        # leaves room above the data-driven p99.9 scale so ``cap > scale``
        # keeps the tanh gradient alive over the full operational envelope.
        #
        # a_spec: ±8 m/s² (~0.8 g) covers emergency braking after landing
        # per FAA acceleration brochure (typical max 0.6-0.8 g).
        "fdm_a_spec_ms2": 8.0,
        # n_z_residual: ±2.0 (n_z certified up to 2.5 g per 14 CFR 25.337,
        # so residual = n_z - 1 in [-2.0, +1.5]; we use the symmetric bound).
        "fdm_n_z_residual": 2.0,
        # phi_bank: pi/3 (60°) — commercial operating limit, FAR/CS-25
        # normal manoeuvres. tan(pi/3) ≈ 1.73 keeps ``g/V·tan(phi)``
        # numerically well-behaved.
        "fdm_phi_bank_rad": math.pi / 3,
    },
    # Compute the data-driven scale on the *active* signal: filter out
    # samples where |x| <= 1% * p999 (cruise / straight flight) so the
    # resulting p99.9 reflects the natural unit of operating manoeuvres
    # rather than being diluted by long stretches of near-zero output.
    nn_output_scale_floor_ratio=0.01,
)

register(NODE_ADSB_V1)
