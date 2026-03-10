"""Tests for the fdm CLI application."""

from __future__ import annotations

import subprocess
import sys


class TestCLI:
    """Tests for CLI entry point and basic commands."""

    def test_cli_help(self) -> None:
        """``python -m node_fdm_pipeline --help`` exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # cyclopts may output to stdout or stderr depending on version
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "fdm" in output.lower() or "pipeline" in output.lower()

    def test_cli_version(self) -> None:
        """``python -m node_fdm_pipeline version`` prints version string."""
        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "node-fdm-pipeline" in output

    def test_cli_train_help(self) -> None:
        """``fdm train --help`` shows train command options."""
        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "train", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--arch" in output
        assert "--config" in output

    def test_cli_predict_help(self) -> None:
        """``fdm predict --help`` shows predict command options."""
        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "predict", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--arch" in output
        assert "--device" in output

    def test_cli_predict_bada_help(self) -> None:
        """``fdm predict-bada --help`` shows BADA command options."""
        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "predict-bada", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--config" in output

    def test_cli_dataset_stats_help(self) -> None:
        """``fdm dataset-stats --help`` shows stats command options."""
        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "dataset-stats", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--arch" in output
