"""Shared test fixtures for node-fdm-pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


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
    config = tmp_path / "config.yaml"
    config.write_text("""\
paths:
  data_dir: "TODO"
  preprocess_dir: "preprocessed_parquet"
  era5_cache_dir: "era5_cache"

era5_features:
  - u_component_of_wind
  - v_component_of_wind
  - temperature

typecodes:
  - A320
  - A20N
  - A321
  - A21N
  - B738
  - B38M
  - E190
  - E195
  - A319
  - A332
  - A333

computing:
  default_cpu_count: 35

bada:
  bada_4_2_dir: "/path/to/BADA/4.2.1"
""")
    return config


@pytest.fixture
def qar_config_path(tmp_path: Path) -> Path:
    """Create a minimal QAR config YAML (replaces deleted scripts/qar/config.yaml)."""
    config = tmp_path / "config.yaml"
    config.write_text("""\
paths:
  data_dir: "TODO"

typecodes:
  - A320
""")
    return config


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal valid config YAML in a temp directory."""
    data_dir = tmp_path / "test_data"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
    )
    return config


@pytest.fixture
def empty_typecodes_config(tmp_path: Path) -> Path:
    """Create a config YAML with empty typecodes (should fail validation)."""
    data_dir = tmp_path / "test_data"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""\
paths:
  data_dir: "{data_dir}"

typecodes: []
"""
    )
    return config
