"""ADS-B v1 architecture specification.

Simplified two-layer architecture (trajectory + data_ode + physics) for
ADS-B data.  Compared to ``opensky_2025``:

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

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm_data.schemas.adsb import DX_COLS, E0_COLS, E1_COLS, U_COLS, U_ODE_COLS, X_COLS

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
            name="data_ode",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=X_COLS
            + U_ODE_COLS
            + E0_COLS
            + E1_COLS
            + [
                "fdm_gamma_target_known",
                "fdm_tas_target_known",
                "fdm_heading_target_known",
                "fdm_heading_known",
            ],
            output_cols=["fdm_a_spec_ms2", "fdm_n_z_residual", "fdm_phi_bank_rad"],
            trainable=True,
            config={
                "denormalize_modes": {
                    "fdm_a_spec_ms2": "scaled",
                    "fdm_n_z_residual": "scaled",
                    "fdm_phi_bank_rad": "scaled",
                },
                # Scales calibrated on real ADS-B distribution (p99.9):
                #   a_spec      : std=0.6,  p99.9=2.6 m/s²
                #   n_z_residual: std=0.022, p99.9=0.13
                # cap = p99.9 (physical bound), scale = cap so denorm = cap*tanh(x).
                # phi_bank: hard cap at 1.0 rad (Decision Q1 from briefing —
                # ~5.5% of empirical samples saturate; tan(1.0)=1.557 keeps the
                # ``g/V·tan`` term well-conditioned and the dx clamp ±0.1 rad/s
                # then bounds the rate physically).
                "scale_overrides": {
                    "fdm_a_spec_ms2": 2.5,
                    "fdm_n_z_residual": 0.13,
                    "fdm_phi_bank_rad": 1.0,
                },
                "cap_overrides": {
                    "fdm_a_spec_ms2": 2.5,
                    "fdm_n_z_residual": 0.13,
                    "fdm_phi_bank_rad": 1.0,
                },
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
)

register(NODE_ADSB_V1)
