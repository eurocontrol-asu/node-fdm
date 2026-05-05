"""Tests for the architecture registry."""

from __future__ import annotations

import pytest

from node_fdm.architectures.registry import (
    ArchitectureSpec,
    LayerSpec,
    get,
    register,
)


class TestRegistry:
    """Architecture registry tests."""

    def test_register_and_get(self) -> None:
        """Register a spec and retrieve it by name."""
        spec = ArchitectureSpec(
            name="test_arch",
            x_cols=["x1"],
            u_cols=["u1"],
            e0_cols=["e0"],
            e1_cols=["e1"],
            dx_cols=[(1, "dx1")],
            layers=[
                LayerSpec(
                    name="layer1",
                    layer_class="node_fdm.layers.blocks.MLPBlock",
                    input_cols=["x1"],
                    output_cols=["dx1"],
                ),
            ],
        )
        register(spec)
        assert get("test_arch") is spec

    def test_get_unknown_raises(self) -> None:
        """Requesting an unknown architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            get("nonexistent_arch_xyz")

    def test_duplicate_registration_warns(self) -> None:
        """Registering the same name twice emits a warning."""
        spec = ArchitectureSpec(
            name="dup_arch",
            x_cols=["x"],
            u_cols=[],
            e0_cols=[],
            e1_cols=[],
            dx_cols=[],
            layers=[],
        )
        register(spec)
        with pytest.warns(UserWarning, match="already registered"):
            register(spec)

    def test_adsb_spec_cols(self) -> None:
        """ADS-B v1 spec has expected column lists and layer composition."""
        import node_fdm.architectures.adsb  # noqa: F401 — triggers auto-register

        spec = get("node_adsb_v1")
        assert spec.x_cols == [
            "raw_alt_m",
            "fdm_gamma_rad",
            "era_tas_ms",
            "fdm_heading_rad",
        ]
        assert len(spec.layers) == 4
        assert spec.layers[0].name == "trajectory"
        assert spec.layers[-1].name == "physics"

    def test_qar_spec_cols(self) -> None:
        """QAR spec has expected column lists."""
        import node_fdm.architectures.qar  # noqa: F401 — triggers auto-register

        spec = get("qar")
        assert len(spec.layers) == 4
        assert spec.layers[0].name == "trajectory"
        assert spec.layers[3].name == "data_ode"


class TestArchitectureSpecBounds:
    """Tests for physical bounds on ArchitectureSpec."""

    def test_spec_with_bounds(self) -> None:
        """ArchitectureSpec accepts x_bounds and dx_bounds fields."""
        x_bounds = {
            "alt_m": (0.0, 15000.0),
            "tas_ms": (50.0, 300.0),
        }
        dx_bounds = {
            "d_alt_ms": (-50.0, 50.0),
        }
        spec = ArchitectureSpec(
            name="bounds_test",
            x_cols=["alt_m", "tas_ms"],
            u_cols=[],
            e0_cols=[],
            e1_cols=[],
            dx_cols=[(1, "d_alt_ms")],
            layers=[],
            x_bounds=x_bounds,
            dx_bounds=dx_bounds,
        )
        assert spec.x_bounds == x_bounds
        assert spec.dx_bounds == dx_bounds
        assert spec.x_bounds["alt_m"] == (0.0, 15000.0)
        assert spec.dx_bounds["d_alt_ms"] == (-50.0, 50.0)

    def test_spec_without_bounds(self) -> None:
        """ArchitectureSpec defaults to empty dicts when bounds omitted."""
        spec = ArchitectureSpec(
            name="no_bounds_test",
            x_cols=["x1"],
            u_cols=[],
            e0_cols=[],
            e1_cols=[],
            dx_cols=[],
            layers=[],
        )
        assert spec.x_bounds == {}
        assert spec.dx_bounds == {}

    def test_adsb_bounds_registered(self) -> None:
        """node_adsb_v1 spec declares physical bounds.

        ``fdm_heading_rad`` is intentionally NOT bounded (Phase 2B
        Decision 4: the projected integrator would clip the wrap-around
        instead of letting it wrap freely).
        """
        import node_fdm.architectures.adsb  # noqa: F401 — triggers auto-register

        spec = get("node_adsb_v1")
        assert len(spec.x_bounds) == 3
        assert "fdm_heading_rad" not in spec.x_bounds
        assert len(spec.dx_bounds) == 4
        assert spec.dx_bounds["fdm_d_heading_rads"] == (-0.1, 0.1)

    def test_partial_bounds(self) -> None:
        """Only some columns have bounds; unbounded columns unaffected."""
        spec = ArchitectureSpec(
            name="partial_bounds_test",
            x_cols=["alt_m", "tas_ms", "gamma_rad"],
            u_cols=[],
            e0_cols=[],
            e1_cols=[],
            dx_cols=[(1, "d_alt_ms"), (1, "d_tas_ms")],
            layers=[],
            x_bounds={"alt_m": (0.0, 15000.0)},
            dx_bounds={"d_alt_ms": (-50.0, 50.0)},
        )
        assert "alt_m" in spec.x_bounds
        assert "tas_ms" not in spec.x_bounds
        assert "gamma_rad" not in spec.x_bounds
        assert "d_alt_ms" in spec.dx_bounds
        assert "d_tas_ms" not in spec.dx_bounds

    def test_qar_no_bounds(self) -> None:
        """QAR architecture loads with empty bounds (no projected integration)."""
        import node_fdm.architectures.qar  # noqa: F401 — triggers auto-register

        qar = get("qar")
        assert qar.x_bounds == {}
        assert qar.dx_bounds == {}
