"""Tests for the fdm CLI application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _config_fixtures import SELECTED_PARAMS_YAML


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

    def test_import_cli_module(self) -> None:
        """Direct import of cli module covers module-level code."""
        from node_fdm_pipeline.cli import app

        assert app is not None
        assert app.name == ("fdm",)

    def test_import_main_module(self) -> None:
        """cli.main is a callable entry point for __main__."""
        from node_fdm_pipeline.cli import main

        assert callable(main)


class TestCLIDataCommands:
    """CLI --help smoke tests for data pipeline commands."""

    @pytest.mark.parametrize(
        ("cmd", "expected_flags"),
        [
            ("preprocess", ["--config", "--dry-run"]),
            ("convert", ["--config", "--dry-run"]),
            ("identify", ["--config", "--dry-run"]),
            ("enrich", ["--config", "--dry-run"]),
            ("derive", ["--config", "--dry-run"]),
            ("segments", ["--config", "--dry-run"]),
            ("split", ["--config", "--dry-run"]),
            ("flag", ["--config", "--dry-run"]),
            ("table-info", ["--table-path"]),
        ],
    )
    def test_data_cmd_help(self, cmd: str, expected_flags: list[str]) -> None:
        """``fdm <cmd> --help`` exits 0 and shows expected flags."""
        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", cmd, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        for flag in expected_flags:
            assert flag in output, f"{flag!r} not in {cmd} --help output"


class TestCLIDataDryRun:
    """CLI dry-run tests for data commands via subprocess."""

    @staticmethod
    def _write_config(tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "aircraft_db.csv").write_text(
            "icao24,registration,typecode,age,airline\nabc123,F-WXYZ,A320,5,AFR\n"
        )
        config = tmp_path / "config.yaml"
        config.write_text(
            f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n' + SELECTED_PARAMS_YAML
        )
        return Path(config)

    @pytest.mark.parametrize(
        "subcommand, extra_args",
        [
            ("download", ["--start-date", "2025-01-01", "--end-date", "2025-01-02"]),
            ("preprocess", []),
            ("convert", []),
        ],
    )
    def test_dry_run_cli(self, tmp_path: Path, subcommand: str, extra_args: list[str]) -> None:
        """``fdm <subcommand> --dry-run`` exits 0."""
        config = self._write_config(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "node_fdm_pipeline",
                subcommand,
                "--config",
                str(config),
                *extra_args,
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
