"""Unit tests for `_resolve_model_path` in commands.predict.

Pure-resolution tests: no model is loaded, no Delta is read.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_resolve_model_path_with_explicit_model_name() -> None:
    """`model_name` overrides the legacy `<info.name>_<acft>` convention."""
    from node_fdm_pipeline.commands.predict import _resolve_model_path

    info = SimpleNamespace(name="node_adsb_v1")
    models_dir = Path("/fake/models")

    result = _resolve_model_path(
        info=info,
        acft="A320",
        local_model=True,
        models_dir=models_dir,
        model_name="node_adsb_v1_A320_06-05",
    )

    assert result == models_dir / "node_adsb_v1_A320_06-05"


def test_resolve_model_path_default_legacy_convention() -> None:
    """Without `model_name`, legacy `<info.name>_<acft>` path is preserved."""
    from node_fdm_pipeline.commands.predict import _resolve_model_path

    info = SimpleNamespace(name="node_adsb_v1")
    models_dir = Path("/fake/models")

    result = _resolve_model_path(
        info=info,
        acft="A320",
        local_model=True,
        models_dir=models_dir,
    )

    assert result == models_dir / "node_adsb_v1_A320"


def test_resolve_model_path_with_model_name_does_not_use_acft() -> None:
    """When `model_name` is set, the typecode is not appended."""
    from node_fdm_pipeline.commands.predict import _resolve_model_path

    info = SimpleNamespace(name="node_adsb_v1")
    models_dir = Path("/fake/models")

    result = _resolve_model_path(
        info=info,
        acft="B737",  # would normally append _B737 to the path
        local_model=True,
        models_dir=models_dir,
        model_name="custom_checkpoint",
    )

    assert result == models_dir / "custom_checkpoint"
    assert "B737" not in str(result)
