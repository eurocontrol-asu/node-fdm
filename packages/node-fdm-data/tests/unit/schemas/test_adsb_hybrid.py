from __future__ import annotations

from node_fdm_data.schemas import adsb, adsb_hybrid


def test_adsb_baseline_schema_untouched() -> None:
    """AC1: baseline ADS-B schema keeps its original 4D state."""
    assert adsb.X_COLS == [
        "raw_alt_m",
        "fdm_gamma_rad",
        "era_tas_ms",
        "fdm_heading_rad",
    ]
    assert len(adsb.DX_COLS) == 4


def test_hybrid_x_cols_extends_baseline_with_mass() -> None:
    """AC3: hybrid state extends baseline ADS-B state with mass last."""
    assert adsb_hybrid.X_COLS == [*adsb.X_COLS, "fdm_mass_kg"]
    assert len(adsb_hybrid.X_COLS) == 5
    assert adsb_hybrid.X_COLS[-1] == "fdm_mass_kg"


def test_hybrid_dx_cols_includes_mass_zero_coefficient() -> None:
    """AC4: hybrid derivative schema adds fixed-mass derivative."""
    assert adsb_hybrid.DX_COLS[-1] == (0, "fdm_d_mass_kgs")
    assert isinstance(adsb_hybrid.DX_COLS[-1][0], int)
    assert len(adsb_hybrid.DX_COLS) == 5


def test_hybrid_reexports_unchanged_lists() -> None:
    """AC5: hybrid schema re-exports unchanged baseline control/environment lists."""
    assert adsb_hybrid.U_COLS == adsb.U_COLS
    assert adsb_hybrid.U_ODE_COLS == adsb.U_ODE_COLS
    assert adsb_hybrid.E0_COLS == adsb.E0_COLS
    assert adsb_hybrid.E1_COLS == adsb.E1_COLS


def test_flight_feature_cols_5_order() -> None:
    """AC6, AC8: 5-feature MassEncoder set keeps the documented order."""
    assert adsb_hybrid.FLIGHT_FEATURE_COLS_5 == [
        "dist_total_flight",
        "dist_adep_at_t0",
        "cruise_alt_max_flight",
        "wind_long_mean_flight",
        "temp_isa_dev_mean_flight",
    ]


def test_flight_feature_cols_6_extends_5_with_mach() -> None:
    """6-feature set is the 5-feature set + mach_cruise_planned at the tail."""
    assert adsb_hybrid.FLIGHT_FEATURE_COLS_6 == [
        *adsb_hybrid.FLIGHT_FEATURE_COLS_5,
        "mach_cruise_planned",
    ]


def test_flight_feature_signs_match_cols() -> None:
    """MassEncoder feature signs match the feature column order for both sets."""
    assert adsb_hybrid.FLIGHT_FEATURE_SIGNS_5 == [1.0, -1.0, -1.0, -1.0, 1.0]
    assert adsb_hybrid.FLIGHT_FEATURE_SIGNS_6 == [1.0, -1.0, -1.0, -1.0, 1.0, -1.0]
    assert len(adsb_hybrid.FLIGHT_FEATURE_SIGNS_5) == len(adsb_hybrid.FLIGHT_FEATURE_COLS_5)
    assert len(adsb_hybrid.FLIGHT_FEATURE_SIGNS_6) == len(adsb_hybrid.FLIGHT_FEATURE_COLS_6)


def test_a320_tcds_bounds_constants() -> None:
    """AC6: A320 TCDS mass bounds are exposed as ordered float constants."""
    assert adsb_hybrid.A320_OEW_KG == 42_600.0
    assert adsb_hybrid.A320_MTOW_KG == 77_000.0
    assert adsb_hybrid.A320_OEW_KG < adsb_hybrid.A320_MTOW_KG


def test_all_exports_complete() -> None:
    """AC7: hybrid schema exports every public constant."""
    assert set(adsb_hybrid.__all__) == {
        "X_COLS",
        "U_COLS",
        "U_ODE_COLS",
        "E0_COLS",
        "E1_COLS",
        "DX_COLS",
        "FLIGHT_FEATURE_COLS_5",
        "FLIGHT_FEATURE_COLS_6",
        "FLIGHT_FEATURE_SIGNS_5",
        "FLIGHT_FEATURE_SIGNS_6",
        "A320_OEW_KG",
        "A320_MTOW_KG",
    }
