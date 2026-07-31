"""Tests for the generic architecture registry and provider contract."""

from __future__ import annotations

from collections.abc import Iterator
from importlib import metadata
from types import SimpleNamespace
from typing import Any

import pytest

from node_fdm.architectures import (
    ArchitectureSpec,
    LayerSpec,
    architecture_digest,
    available,
    discover_architectures,
    get,
    get_origin,
    register,
    registry,
)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test isolated state without corrupting later test modules."""
    previous_registry = dict(registry.REGISTRY)
    previous_origins = dict(registry.ORIGINS)
    previous_complete = registry._DISCOVERY_COMPLETE
    registry.REGISTRY.clear()
    registry.ORIGINS.clear()
    monkeypatch.setattr(registry, "_DISCOVERY_COMPLETE", False)
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: [])
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(previous_registry)
    registry.ORIGINS.clear()
    registry.ORIGINS.update(previous_origins)
    registry._DISCOVERY_COMPLETE = previous_complete


def make_spec(name: str = "test_arch") -> ArchitectureSpec:
    """Build the smallest valid architecture for registry tests."""
    return ArchitectureSpec(
        name=name,
        x_cols=["x"],
        u_cols=[],
        e0_cols=[],
        e1_cols=[],
        dx_cols=[(1, "dx")],
        layers=[
            LayerSpec(
                name="layer",
                layer_class="node_fdm.layers.blocks.MLPBlock",
                input_cols=["x"],
                output_cols=["dx"],
            )
        ],
    )


def test_register_and_get_direct_spec() -> None:
    spec = make_spec()
    register(spec)
    assert get(spec.name) is spec


def test_duplicate_direct_registration_warns() -> None:
    spec = make_spec()
    register(spec)
    with pytest.warns(UserWarning, match="already registered"):
        register(spec)


def test_discovery_registers_canonical_name_alias_and_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_spec("provider_arch")

    class FakeEntryPoint:
        name = "example"
        value = "example_models:architectures"
        dist = SimpleNamespace(name="example-models", version="1.2.3")

        @staticmethod
        def load() -> Any:
            return lambda: {"friendly": spec}

    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: [FakeEntryPoint()])
    discover_architectures()

    assert get("friendly") is spec
    assert get("provider_arch") is spec
    assert available() == ("friendly", "provider_arch")
    origin = get_origin("friendly")
    assert origin is not None
    assert origin.distribution == "example-models"
    assert origin.version == "1.2.3"


def test_provider_collision_fails_instead_of_overwriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = make_spec("shared")
    register(direct)
    other = make_spec("other")

    class FakeEntryPoint:
        name = "conflicting"
        value = "conflicting_models:architectures"
        dist = None

        @staticmethod
        def load() -> Any:
            return lambda: {"shared": other}

    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: [FakeEntryPoint()])
    with pytest.raises(ValueError, match="collides"):
        discover_architectures()
    assert registry.REGISTRY["shared"] is direct


def test_architecture_digest_is_stable_and_sensitive() -> None:
    first = make_spec("digest")
    same = ArchitectureSpec.model_validate(first.model_dump())
    changed = first.model_copy(update={"x_cols": ["different"]})

    assert architecture_digest(first) == architecture_digest(same)
    assert architecture_digest(first) != architecture_digest(changed)
