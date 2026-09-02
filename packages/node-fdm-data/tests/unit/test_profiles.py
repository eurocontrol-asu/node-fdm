from __future__ import annotations

import pytest
from pydantic import ValidationError

import node_fdm_data


def test_opensky26_exp03_channels_have_exact_parameters() -> None:
    """AC1: the five channels carry the exact selected experiment parameters."""
    profile = node_fdm_data.load_profile("opensky26-exp03-v1")

    expected = {
        "alt": (5, 20, 3.6342411857, 30.48, 8),
        "gamma": (0.002, 4, 0.000144225, 0.002, 10),
        "vz": (50, 4, 21.9381500315, 100, 8),
        "mach": (0.0025, 8, 0.0003162278, 0.05, 15),
        "cas": (1.5, 8, 0.3534391546, 20, 5),
    }

    actual = {
        name: (
            channel.sigma_r,
            channel.sigma_s,
            channel.slope_tol,
            channel.flat_tol,
            channel.min_len,
        )
        for name, channel in profile.channels.items()
    }
    assert actual == expected


def test_opensky26_exp03_provenance_is_exact() -> None:
    """AC2: provenance retains the source commit and all three source digests."""
    profile = node_fdm_data.load_profile("opensky26-exp03-v1")

    assert profile.provenance.commit == "80e8039f85a59dc214630386ca637ea7a296cfa3"
    assert profile.provenance.sha256 == {
        "retained.csv": "2654c9b688669722271246868633dca0598cc345651e54feb1245affa35b58e1",
        "grids.json": "55a80354047a8d5c61f41921b4541f98699e9a158db5c427a99a9d6f5c288805",
        "metrics.yaml": "3c14d9a77705511296121f63bf2d957679b200d23e63b150a5f0d8ffc54a0a14",
    }


def test_opensky26_exp03_declares_bilateral_family_rules() -> None:
    """AC3: the profile declares the complete bilateral two-pass configuration."""
    profile = node_fdm_data.load_profile("opensky26-exp03-v1")

    assert profile.family == "bilateral"
    assert profile.passes == 2
    assert profile.butter == 0
    assert profile.alt_gate is False
    assert profile.mach_floor is None
    assert profile.prefilter is None
    assert profile.vertical_cascade == ("alt", "gamma", "vz")
    assert profile.speed_cascade == ("mach", "cas")
    assert profile.exclusion == "post_detection"
    assert profile.frozen_endpoint.columns == (
        "altitude",
        "ground_speed",
        "track",
        "latitude",
        "longitude",
    )
    assert profile.frozen_endpoint.min_samples == 15
    assert profile.channels["gamma"].abs_min is None
    assert profile.channels["vz"].min_abs is None


def test_loaded_profile_and_channels_are_immutable() -> None:
    """AC4: both the profile and each nested channel reject attribute assignment."""
    profile = node_fdm_data.load_profile("opensky26-exp03-v1")

    with pytest.raises(ValidationError):
        profile.passes = 3
    with pytest.raises(ValidationError):
        profile.channels["alt"].min_len = 9


def test_profile_registry_lists_known_name_and_rejects_unknown_name() -> None:
    """AC5: registry discovery includes the profile and unknown lookup raises KeyError."""
    assert "opensky26-exp03-v1" in node_fdm_data.list_profiles()

    with pytest.raises(KeyError):
        node_fdm_data.load_profile("does-not-exist")
