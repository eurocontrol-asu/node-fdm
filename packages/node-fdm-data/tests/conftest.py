"""Shared fixtures for node-fdm-data tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from node_fdm_pipeline.config import SelectedParamConfig


@pytest.fixture
def segment_config_full() -> dict[str, Any]:
    """Full segment-detection config covering mach/cas/vz/gamma/alt.

    Sections are independent: missing input columns simply skip the
    corresponding output, so the full config is safe to use even when a
    test only exercises a subset of parameters.
    """
    return {
        "mach": {
            "tol": 0.0005,
            "min_len": 30,
            "alt_threshold": 15000,
            "smooth_window": 10,
            "use_alt": True,
        },
        "cas": {
            "tol": 0.75,
            "min_len": 20,
            "use_alt": False,
            "smooth_window": 10,
            "smooth_method": "savgol",
        },
        "vz": {
            "tol": 25,
            "min_len": 20,
            "use_alt": False,
            "min_abs_value": 75,
            "smooth_window": 15,
            "smooth_method": "savgol",
        },
        "gamma": {
            "tol": 0.002,
            "min_len": 15,
            "use_alt": False,
            "smooth_window": 5,
            "smooth_method": "savgol",
        },
        "alt": {
            "tol": 25,
            "min_len": 5,
            "use_alt": False,
            "min_abs_value": 25,
            "smooth_window": 5,
            "smooth_method": "savgol",
        },
    }


#: opensky26 tbl.3 detector tuning, for tests that build a
#: ``node_fdm_pipeline.config`` filter model. Those hyper-parameters are
#: required (a calibration result, not a constant), so a test constructing one
#: must state which calibration it is running.
SELECTED_PARAMS: dict[str, dict[str, float | int]] = {
    "mach": {
        "sigma_s": 8.0,
        "sigma_r": 0.01,
        "n_passes": 2,
        "slope_tol": 3.0e-4,
        "flat_tol": 5.0e-2,
        "min_len": 15,
    },
    "cas": {
        "cutoff_s": 180.0,
        "sigma_s": 8.0,
        "sigma_r": 3.0,
        "n_passes": 2,
        "slope_tol": 0.09,
        "flat_tol": 20.0,
        "min_len": 5,
    },
    "vz": {
        "sigma_s": 6.0,
        "sigma_r": 100.0,
        "slope_tol": 50.0,
        "flat_tol": 100.0,
        "min_len": 10,
    },
    "alt": {
        "sigma_s": 6.0,
        "sigma_r": 20.0,
        "n_passes": 2,
        "tol_ftmin": 150.0,
        "min_len": 6,
    },
    "gamma": {
        "sigma_s": 6.0,
        "sigma_r": 0.002,
        "slope_tol": 1.2e-3,
        "flat_tol": 2.0e-3,
        "abs_min": 5.0e-3,
        "min_len": 10,
    },
}


def selected_param_config() -> SelectedParamConfig:
    """Build a fully-declared ``SelectedParamConfig`` from the block above.

    Mypy cannot follow ``SelectedParamConfig(**SELECTED_PARAMS)`` — the nested
    dicts are not the per-channel model types it expects — so the construction
    is written out once here instead of being ignored at each call site.
    """
    from node_fdm_pipeline.config import (
        AltFilterConfig,
        CasFilterConfig,
        GammaFilterConfig,
        MachFilterConfig,
        SelectedParamConfig,
        VzFilterConfig,
    )

    return SelectedParamConfig(
        mach=MachFilterConfig(**SELECTED_PARAMS["mach"]),  # type: ignore[arg-type]
        cas=CasFilterConfig(**SELECTED_PARAMS["cas"]),  # type: ignore[arg-type]
        vz=VzFilterConfig(**SELECTED_PARAMS["vz"]),  # type: ignore[arg-type]
        alt=AltFilterConfig(**SELECTED_PARAMS["alt"]),  # type: ignore[arg-type]
        gamma=GammaFilterConfig(**SELECTED_PARAMS["gamma"]),  # type: ignore[arg-type]
    )


@pytest.fixture
def selected_params() -> dict[str, dict[str, float | int]]:
    """The required detector tuning as nested dicts (opensky26 tbl.3)."""
    return SELECTED_PARAMS


@pytest.fixture
def selected_param_config_factory() -> Callable[[], SelectedParamConfig]:
    """A fully-declared ``SelectedParamConfig`` builder (opensky26 tbl.3)."""
    return selected_param_config
