"""Unit tests for CLI entry points (no I/O)."""

from __future__ import annotations

import pytest


def test_version_cmd(capsys: pytest.CaptureFixture[str]) -> None:
    """version_cmd prints version string."""
    from node_fdm_pipeline.cli import version_cmd

    version_cmd()
    captured = capsys.readouterr()
    assert "node-fdm-pipeline" in captured.out
