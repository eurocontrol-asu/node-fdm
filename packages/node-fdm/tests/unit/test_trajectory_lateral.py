"""Lateral channel formulas inside ``TrajectoryLayer`` (Phase 2B).

Validates:

1. The 2D ground-speed upgrade: ``gs = sqrt((tas·sinψ + u)² + (tas·cosψ + v)²)``.
2. The drift formula: ``drift = atan2(u·cosψ - v·sinψ, tas + u·sinψ + v·cosψ)``.
3. The wrap-aware ``heading_diff = known · signed_wrap(target - heading)``.

The convention for the wind decomposition is the one published by the
data layer (Phase 2A): ``along = u·sinψ + v·cosψ`` (heading rotates
clockwise from north, ψ=0 ⇒ north ⇒ ``along = v``).
"""

from __future__ import annotations

import math

import pytest
import torch

from node_fdm.layers.trajectory import TrajectoryLayer

# Mapping wires the canonical lateral keys to plain test column names so the
# assertions below stay readable.
COL_MAP = {
    "tas": "tas",
    "gamma": "gamma",
    "alt": "alt",
    "wind": "wind",
    "alt_sel": "alt_sel",
    "vz": "vz",
    "gs": "gs",
    "mach": "mach",
    "cas": "cas",
    "alt_diff": "alt_diff",
    "g_sin_gamma": "g_sin_gamma",
    "cos_gamma": "cos_gamma",
    "q": "q",
    "g_over_v": "g_over_v",
    "u_wind": "u",
    "v_wind": "v",
    "heading": "heading",
    "heading_target": "heading_target",
    "heading_target_known": "heading_target_known",
    "heading_known": "heading_known",
    "lat_wind": "lat_wind",
    "drift": "drift",
    "track": "track",
    "heading_diff": "heading_diff",
}


def _base_inputs(
    *,
    heading: float,
    tas: float = 100.0,
    u: float = 0.0,
    v: float = 0.0,
    alt: float = 5000.0,
    gamma: float = 0.0,
) -> dict[str, torch.Tensor]:
    return {
        "tas": torch.tensor([tas]),
        "gamma": torch.tensor([gamma]),
        "alt": torch.tensor([alt]),
        "wind": torch.tensor([0.0]),
        "u": torch.tensor([u]),
        "v": torch.tensor([v]),
        "heading": torch.tensor([heading]),
    }


class TestLateralGroundSpeed:
    """2D ground-speed norm replaces the longitudinal scalar."""

    def test_north_no_wind(self) -> None:
        """ψ=0 (north), no wind ⇒ gs == tas, drift == 0."""
        layer = TrajectoryLayer(col_map=COL_MAP)
        out = layer.forward(_base_inputs(heading=0.0))
        assert math.isclose(out["gs"].item(), 100.0, rel_tol=1e-5)
        assert abs(out["drift"].item()) < 1e-6

    def test_north_with_eastern_crosswind(self) -> None:
        """ψ=0, u=20 m/s east cross-wind ⇒ gs=√(100²+20²), drift=atan(20/100)."""
        layer = TrajectoryLayer(col_map=COL_MAP)
        out = layer.forward(_base_inputs(heading=0.0, u=20.0))
        assert math.isclose(out["gs"].item(), math.sqrt(100**2 + 20**2), rel_tol=1e-5)
        assert math.isclose(out["drift"].item(), math.atan2(20.0, 100.0), rel_tol=1e-5)

    def test_east_with_eastern_tailwind(self) -> None:
        """ψ=π/2 (east), u=20 east ⇒ pure tail wind ⇒ gs=120, drift≈0."""
        layer = TrajectoryLayer(col_map=COL_MAP)
        out = layer.forward(_base_inputs(heading=math.pi / 2, u=20.0))
        assert math.isclose(out["gs"].item(), 120.0, rel_tol=1e-5)
        assert abs(out["drift"].item()) < 1e-6


class TestHeadingDiff:
    """Wrap-aware ``heading_diff = known · signed_wrap(target - heading)``."""

    def test_no_target_returns_no_diff(self) -> None:
        """Without a target column, no heading_diff is published."""
        layer = TrajectoryLayer(col_map=COL_MAP)
        out = layer.forward(_base_inputs(heading=0.0))
        assert "heading_diff" not in out

    def test_diff_simple(self) -> None:
        """target=0.5, heading=0.0 ⇒ diff=+0.5 when known=1."""
        layer = TrajectoryLayer(col_map=COL_MAP)
        x = _base_inputs(heading=0.0)
        x["heading_target"] = torch.tensor([0.5])
        x["heading_target_known"] = torch.tensor([1.0])
        out = layer.forward(x)
        assert math.isclose(out["heading_diff"].item(), 0.5, rel_tol=1e-5)

    def test_diff_wrap_around(self) -> None:
        """target=π-0.05, heading=-π+0.05 ⇒ diff=-0.10 (not 2π-0.10)."""
        layer = TrajectoryLayer(col_map=COL_MAP)
        x = _base_inputs(heading=-math.pi + 0.05)
        x["heading_target"] = torch.tensor([math.pi - 0.05])
        x["heading_target_known"] = torch.tensor([1.0])
        out = layer.forward(x)
        assert math.isclose(out["heading_diff"].item(), -0.10, abs_tol=1e-5)

    def test_diff_gated_by_known(self) -> None:
        """known=0 ⇒ heading_diff = 0 regardless of target."""
        layer = TrajectoryLayer(col_map=COL_MAP)
        x = _base_inputs(heading=0.0)
        x["heading_target"] = torch.tensor([1.5])
        x["heading_target_known"] = torch.tensor([0.0])
        out = layer.forward(x)
        assert abs(out["heading_diff"].item()) < 1e-9


@pytest.mark.parametrize(
    "heading_deg,u,v,expected_drift_deg",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 10.0, 0.0, math.degrees(math.atan2(10.0, 100.0))),
        (90.0, 0.0, 10.0, math.degrees(math.atan2(-10.0, 100.0))),
    ],
)
def test_drift_parametrised(
    heading_deg: float, u: float, v: float, expected_drift_deg: float
) -> None:
    """A handful of analytic drift cases to anchor the wind-frame convention."""
    layer = TrajectoryLayer(col_map=COL_MAP)
    out = layer.forward(_base_inputs(heading=math.radians(heading_deg), u=u, v=v))
    assert math.isclose(math.degrees(out["drift"].item()), expected_drift_deg, abs_tol=1e-3)
