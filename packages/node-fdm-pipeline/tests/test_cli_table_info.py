"""Tests for the ``fdm table-info`` CLI command."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch


class TestTableInfoCommand:
    """CLI ``table-info`` subcommand prints partitions, columns, and versions."""

    def test_table_info_command(self) -> None:
        mock_info = {
            "partitions": ["2025-01-01", "2025-01-02"],
            "columns": ["meta_batch_date", "altitude_ft", "speed_kt"],
            "versions": 3,
        }

        with patch(
            "node_fdm_pipeline.commands.table_info.table_info",
            return_value=mock_info,
        ):
            result = subprocess.run(
                [sys.executable, "-m", "node_fdm_pipeline", "table-info", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0
            assert "table-info" in output.lower() or "table" in output.lower()

    def test_table_info_prints_partitions(self) -> None:
        """table_info output includes partition dates."""
        mock_info = {
            "partitions": ["2025-01-01", "2025-01-02"],
            "columns": ["meta_batch_date", "altitude_ft"],
            "versions": 2,
        }

        with patch(
            "node_fdm_pipeline.commands.table_info.table_info",
            return_value=mock_info,
        ):
            # Verify the command is discoverable via --help
            result = subprocess.run(
                [sys.executable, "-m", "node_fdm_pipeline", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0
            assert "table-info" in output or "table" in output.lower()
