"""Tests for the fdm CLI application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


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
        config.write_text(f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n')
        return Path(config)

    def test_download_dry_run_cli(self, tmp_path: Path) -> None:
        """``fdm download --dry-run`` exits 0."""
        config = self._write_config(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "node_fdm_pipeline",
                "download",
                "--config",
                str(config),
                "--start-date",
                "2025-01-01",
                "--end-date",
                "2025-01-02",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_preprocess_dry_run_cli(self, tmp_path: Path) -> None:
        """``fdm preprocess --dry-run`` exits 0."""
        config = self._write_config(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "node_fdm_pipeline",
                "preprocess",
                "--config",
                str(config),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_convert_dry_run_cli(self, tmp_path: Path) -> None:
        """``fdm convert --dry-run`` exits 0."""
        config = self._write_config(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "node_fdm_pipeline",
                "convert",
                "--config",
                str(config),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


class TestCLIDirectInvoke:
    """Direct invocation of CLI wrappers for in-process coverage."""

    @staticmethod
    def _make_config(tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "aircraft_db.csv").write_text(
            "icao24,registration,typecode,age,airline\nabc123,F-WXYZ,A320,5,AFR\n"
        )
        config = tmp_path / "config.yaml"
        config.write_text(f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n')
        return config

    def test_version_cmd(self, capsys: pytest.CaptureFixture[str]) -> None:
        """version_cmd prints version string."""
        from node_fdm_pipeline.cli import version_cmd

        version_cmd()
        captured = capsys.readouterr()
        assert "node-fdm-pipeline" in captured.out

    def test_download_wrapper(self, tmp_path: Path) -> None:
        """CLI download wrapper delegates to commands.data.download."""
        from node_fdm_pipeline.cli import download as download_cmd

        config = self._make_config(tmp_path)
        download_cmd(
            config=config,
            start_date="2025-01-01",
            end_date="2025-01-02",
            dry_run=True,
        )

    def test_preprocess_wrapper(self, tmp_path: Path) -> None:
        """CLI preprocess wrapper delegates to commands.data.preprocess."""
        from node_fdm_pipeline.cli import preprocess as preprocess_cmd

        config = self._make_config(tmp_path)
        preprocess_cmd(config=config, dry_run=True)

    def test_convert_wrapper(self, tmp_path: Path) -> None:
        """CLI convert wrapper delegates to commands.data.convert."""
        from node_fdm_pipeline.cli import convert as convert_cmd

        config = self._make_config(tmp_path)
        convert_cmd(config=config, dry_run=True)

    def test_identify_wrapper(self, tmp_path: Path) -> None:
        """CLI identify wrapper delegates to commands.data.identify."""
        from node_fdm_pipeline.cli import identify as identify_cmd

        config = self._make_config(tmp_path)
        identify_cmd(config=config, dry_run=True)

    def test_flag_wrapper(self, tmp_path: Path) -> None:
        """CLI flag wrapper delegates to commands.data.flag."""
        from node_fdm_pipeline.cli import flag as flag_cmd

        config = self._make_config(tmp_path)
        flag_cmd(config=config, dry_run=True)

    def test_enrich_wrapper(self, tmp_path: Path) -> None:
        """CLI enrich wrapper delegates to commands.data.enrich."""
        from node_fdm_pipeline.cli import enrich as enrich_cmd

        config = self._make_config(tmp_path)
        enrich_cmd(config=config, dry_run=True)

    def test_derive_wrapper(self, tmp_path: Path) -> None:
        """CLI derive wrapper delegates to commands.data.derive."""
        from node_fdm_pipeline.cli import derive as derive_cmd

        config = self._make_config(tmp_path)
        derive_cmd(config=config, dry_run=True)

    def test_segments_wrapper(self, tmp_path: Path) -> None:
        """CLI segments wrapper delegates to commands.data.segments."""
        from node_fdm_pipeline.cli import segments as segments_cmd

        config = self._make_config(tmp_path)
        segments_cmd(config=config, dry_run=True)

    def test_split_wrapper(self, tmp_path: Path) -> None:
        """CLI split wrapper delegates to commands.data.split."""
        from node_fdm_pipeline.cli import split as split_cmd

        config = self._make_config(tmp_path)
        split_cmd(config=config, dry_run=True)

    def test_aircraft_list_wrapper(self, tmp_path: Path) -> None:
        """CLI aircraft-list wrapper delegates to commands.data.aircraft_list."""
        from node_fdm_pipeline.cli import aircraft_list_cmd

        config = self._make_config(tmp_path)
        aircraft_list_cmd(config=config, dry_run=True)

    def test_table_info_wrapper(self, tmp_path: Path) -> None:
        """CLI table-info wrapper delegates to commands.table_info."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import table_info_cmd

        mock_info = {"partitions": ["20250101"], "columns": ["raw_icao24"], "versions": 1}
        with patch(
            "node_fdm_pipeline.commands.table_info.table_info",
            return_value=mock_info,
        ):
            table_info_cmd(table_path=tmp_path / "fake.delta")

    def test_train_wrapper(self, tmp_path: Path) -> None:
        """CLI train wrapper delegates to commands.train.run_training."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import train

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.train.run_training"):
            train(arch="adsb", config=config)

    def test_predict_wrapper(self, tmp_path: Path) -> None:
        """CLI predict wrapper delegates to commands.predict.run_predict."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import predict

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.predict.run_predict"):
            predict(arch="adsb", config=config)

    def test_predict_bada_wrapper(self, tmp_path: Path) -> None:
        """CLI predict-bada wrapper delegates to commands.predict.run_predict_bada."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import predict_bada

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.predict.run_predict_bada"):
            predict_bada(config=config)

    def test_evaluate_wrapper(self, tmp_path: Path) -> None:
        """CLI evaluate wrapper delegates to commands.evaluate.run_evaluate."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import evaluate

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.evaluate.run_evaluate"):
            evaluate(arch="adsb", config=config)

    def test_resume_wrapper(self, tmp_path: Path) -> None:
        """CLI resume wrapper delegates to commands.resume.run_resume."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import resume

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.resume.run_resume"):
            resume(model=model_dir, config=config)

    def test_dataset_stats_wrapper(self, tmp_path: Path) -> None:
        """CLI dataset-stats wrapper delegates to commands.stats."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import dataset_stats

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.stats.run_dataset_stats"):
            dataset_stats(arch="adsb", config=config)

    def test_visualize_wrapper(self, tmp_path: Path) -> None:
        """CLI visualize wrapper delegates to commands.visualize."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import visualize

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.visualize.run_visualize"):
            visualize(arch="adsb", config=config)

    def test_plot_performance_wrapper(self, tmp_path: Path) -> None:
        """CLI plot-performance wrapper delegates to commands.visualize."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import plot_performance

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.visualize.run_plot_performance"):
            plot_performance(config=config)

    def test_plot_example_wrapper(self, tmp_path: Path) -> None:
        """CLI plot-example wrapper delegates to commands.visualize."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import plot_example

        config = self._make_config(tmp_path)
        with patch("node_fdm_pipeline.commands.visualize.run_plot_example"):
            plot_example(config=config)
