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

    def test_opensky_spec_cols(self) -> None:
        """OpenSky 2025 spec has expected column lists."""
        import node_fdm.architectures.opensky  # noqa: F401 — triggers auto-register

        spec = get("opensky_2025")
        assert spec.x_cols == [
            "distance_m",
            "altitude_ft",
            "gamma_rad",
            "tas_kt",
        ]
        assert len(spec.layers) == 2
        assert spec.layers[0].name == "trajectory"
        assert spec.layers[1].name == "data_ode"

    def test_qar_spec_cols(self) -> None:
        """QAR spec has expected column lists."""
        import node_fdm.architectures.qar  # noqa: F401 — triggers auto-register

        spec = get("qar")
        assert len(spec.layers) == 4
        assert spec.layers[0].name == "trajectory"
        assert spec.layers[3].name == "data_ode"
