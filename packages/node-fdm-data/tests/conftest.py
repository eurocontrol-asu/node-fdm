"""Shared fixtures for node-fdm-data tests."""

from __future__ import annotations

from typing import Any

import pytest


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
