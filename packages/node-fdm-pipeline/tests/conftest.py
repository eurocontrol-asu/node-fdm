"""Shared test fixtures for node-fdm-pipeline tests."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from _config_fixtures import (
    SELECTED_PARAMS,
    SELECTED_PARAMS_YAML,
    config_yaml,
    write_config,
)

__all__ = [
    "SELECTED_PARAMS",
    "SELECTED_PARAMS_YAML",
    "config_yaml",
    "write_config",
]


# ``--import-mode=importlib`` keeps the tests dir off sys.path, so test modules
# cannot ``from conftest import ...``. Fixtures are the supported channel.


@pytest.fixture
def make_config_yaml() -> Callable[..., str]:
    """Expose :func:`config_yaml` to test modules."""
    return config_yaml


@pytest.fixture
def make_config() -> Callable[..., Path]:
    """Expose :func:`write_config` to test modules."""
    return write_config


@pytest.fixture
def selected_params_yaml() -> str:
    """The required ``selected_params:`` block as YAML text."""
    return SELECTED_PARAMS_YAML


@pytest.fixture
def selected_params() -> dict[str, dict[str, float | int]]:
    """The required ``selected_params`` values as nested dicts."""
    return SELECTED_PARAMS


@pytest.fixture
def mock_fastmeteo(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Isolated fastmeteo sys.modules mock with strict per-test cleanup.

    Uses ``monkeypatch.setitem`` instead of ``mocker.patch.dict`` so that
    each test gets fresh mocks and ``sys.modules`` is fully restored on
    teardown — preventing cross-test import cache contamination.

    Returns ``(mock_arco_cls, mock_arco_instance)`` so tests can configure
    ``mock_arco_instance.interpolate.side_effect`` as needed.
    """
    mock_arco_cls = MagicMock()
    mock_arco_instance = MagicMock()
    mock_arco_cls.return_value = mock_arco_instance

    mock_source_arco = MagicMock(ArcoEra5=mock_arco_cls)
    mock_source = MagicMock(arco_era5=mock_source_arco)

    monkeypatch.setitem(sys.modules, "fastmeteo", MagicMock())
    monkeypatch.setitem(sys.modules, "fastmeteo.source", mock_source)
    monkeypatch.setitem(sys.modules, "fastmeteo.source.arco_era5", mock_source_arco)

    return mock_arco_cls, mock_arco_instance


@pytest.fixture
def opensky_config_path(tmp_path: Path) -> Path:
    """Create a realistic OpenSky config YAML (replaces deleted scripts/opensky/config.yaml)."""
    return write_config(
        tmp_path / "config.yaml",
        "TODO",
        typecodes=[
            "A320",
            "A20N",
            "A321",
            "A21N",
            "B738",
            "B38M",
            "E190",
            "E195",
            "A319",
            "A332",
            "A333",
        ],
        extra="""\
era5_features:
  - u_component_of_wind
  - v_component_of_wind
  - temperature

computing:
  default_cpu_count: 35

bada:
  bada_4_2_dir: "/path/to/BADA/4.2.1"
""",
    )


@pytest.fixture
def qar_config_path(tmp_path: Path) -> Path:
    """Create a minimal QAR config YAML (replaces deleted scripts/qar/config.yaml)."""
    return write_config(tmp_path / "config.yaml", "TODO")


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal valid config YAML in a temp directory."""
    return write_config(tmp_path / "config.yaml", tmp_path / "test_data")


@pytest.fixture
def empty_typecodes_config(tmp_path: Path) -> Path:
    """Create a config YAML with empty typecodes (should fail validation)."""
    return write_config(tmp_path / "config.yaml", tmp_path / "test_data", typecodes=[])
