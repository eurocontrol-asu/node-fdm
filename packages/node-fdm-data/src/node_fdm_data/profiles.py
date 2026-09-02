"""Versioned, immutable segment-detection profiles."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ChannelParams",
    "FrozenEndpointRule",
    "ProfileProvenance",
    "SegmentProfile",
    "list_profiles",
    "load_profile",
]


class ChannelParams(BaseModel):
    """Selected parameters for one detection channel."""

    model_config = ConfigDict(frozen=True)

    sigma_r: float
    sigma_s: float
    slope_tol: float
    flat_tol: float
    min_len: int
    abs_min: float | None = None
    min_abs: float | None = None


class ProfileProvenance(BaseModel):
    """Opaque identifiers for the source artifacts behind a profile."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    commit: str
    sha256: MappingProxyType[str, str]


class FrozenEndpointRule(BaseModel):
    """Columns and sample threshold used by the frozen-endpoint rule."""

    model_config = ConfigDict(frozen=True)

    columns: tuple[str, ...]
    min_samples: int


class SegmentProfile(BaseModel):
    """Complete immutable configuration for segment detection."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    family: Literal["bilateral"]
    passes: int
    butter: int
    alt_gate: bool
    mach_floor: float | None
    prefilter: str | None
    vertical_cascade: tuple[str, ...]
    speed_cascade: tuple[str, ...]
    exclusion: Literal["post_detection"]
    frozen_endpoint: FrozenEndpointRule
    channels: MappingProxyType[str, ChannelParams]
    provenance: ProfileProvenance


_OPENSKY26_EXP03_V1 = SegmentProfile(
    family="bilateral",
    passes=2,
    butter=0,
    alt_gate=False,
    mach_floor=None,
    prefilter=None,
    vertical_cascade=("alt", "gamma", "vz"),
    speed_cascade=("mach", "cas"),
    exclusion="post_detection",
    frozen_endpoint=FrozenEndpointRule(
        columns=("altitude", "ground_speed", "track", "latitude", "longitude"),
        min_samples=15,
    ),
    channels=MappingProxyType(
        {
            "alt": ChannelParams(
                sigma_r=5,
                sigma_s=20,
                slope_tol=3.6342411857,
                flat_tol=30.48,
                min_len=8,
            ),
            "gamma": ChannelParams(
                sigma_r=0.002,
                sigma_s=4,
                slope_tol=0.000144225,
                flat_tol=0.002,
                min_len=10,
                abs_min=None,
            ),
            "vz": ChannelParams(
                sigma_r=50,
                sigma_s=4,
                slope_tol=21.9381500315,
                flat_tol=100,
                min_len=8,
                min_abs=None,
            ),
            "mach": ChannelParams(
                sigma_r=0.0025,
                sigma_s=8,
                slope_tol=0.0003162278,
                flat_tol=0.05,
                min_len=15,
            ),
            "cas": ChannelParams(
                sigma_r=1.5,
                sigma_s=8,
                slope_tol=0.3534391546,
                flat_tol=20,
                min_len=5,
            ),
        }
    ),
    provenance=ProfileProvenance(
        commit="80e8039f85a59dc214630386ca637ea7a296cfa3",
        sha256=MappingProxyType(
            {
                "retained.csv": (
                    "2654c9b688669722271246868633dca0598cc345651e54feb1245affa35b58e1"
                ),
                "grids.json": ("55a80354047a8d5c61f41921b4541f98699e9a158db5c427a99a9d6f5c288805"),
                "metrics.yaml": (
                    "3c14d9a77705511296121f63bf2d957679b200d23e63b150a5f0d8ffc54a0a14"
                ),
            }
        ),
    ),
)

_PROFILES = MappingProxyType({"opensky26-exp03-v1": _OPENSKY26_EXP03_V1})


def load_profile(name: str) -> SegmentProfile:
    """Return the registered profile named *name*.

    Raises:
        KeyError: If *name* is not registered.
    """
    try:
        return _PROFILES[name]
    except KeyError:
        known_names = ", ".join(_PROFILES)
        raise KeyError(f"Unknown profile {name!r}; known profiles: {known_names}") from None


def list_profiles() -> tuple[str, ...]:
    """Return the registered profile names in deterministic order."""
    return tuple(sorted(_PROFILES))
