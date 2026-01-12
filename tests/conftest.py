"""Shared pytest fixtures for node-fdm tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = Path(__file__).parent / "golden" / "outputs"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def golden_dir() -> Path:
    """Return path to golden outputs directory."""
    return GOLDEN_DIR
