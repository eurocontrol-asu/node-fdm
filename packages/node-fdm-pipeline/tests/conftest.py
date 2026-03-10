"""Shared test fixtures for node-fdm-pipeline tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def opensky_config_path() -> Path:
    """Path to the OpenSky config YAML in the scripts directory."""
    return Path(__file__).resolve().parents[3] / "scripts" / "opensky" / "config.yaml"


@pytest.fixture
def qar_config_path() -> Path:
    """Path to the QAR config YAML in the scripts directory."""
    return Path(__file__).resolve().parents[3] / "scripts" / "qar" / "config.yaml"


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
