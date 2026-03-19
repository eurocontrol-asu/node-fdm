"""Tests for visualization commands."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from node_fdm_pipeline.commands.visualize import (
    _require_viz,
    run_plot_example,
    run_plot_performance,
    run_visualize,
)


class TestRequireViz:
    """Tests for the ``_require_viz`` dependency gate."""

    def test_require_viz_missing_deps(self) -> None:
        """SystemExit raised when viz deps are missing."""
        with (
            patch.dict(
                sys.modules,
                {"matplotlib": None, "altair": None},
            ),
            pytest.raises(SystemExit, match="pip install node-fdm-pipeline"),
        ):
            _require_viz()


class TestRunVisualize:
    """Tests for ``run_visualize``."""

    def _make_config_and_data(self, tmp_path: Path) -> Path:
        """Create config, dirs, and test flight prediction/bada parquets."""
        data_dir = tmp_path / "data"
        predict_dir = data_dir / "predicted_flights" / "A320"
        bada_dir = data_dir / "bada_flights" / "A320"
        figure_dir = data_dir / "figures"
        predict_dir.mkdir(parents=True)
        bada_dir.mkdir(parents=True)
        figure_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        n = 20

        # Create pred and bada parquet (flight_id = "flight001")
        pred_data = pl.DataFrame(
            {
                "pred_raw_alt_m": [10050.0] * n,
                "pred_era_tas_ms": [201.0] * n,
                "pred_fdm_gamma_rad": [0.011] * n,
            }
        )
        pred_data.write_parquet(predict_dir / "flight001.parquet")

        bada_data = pl.DataFrame(
            {
                "bada_alt_std_m": [10020.0] * n,
                "bada_tas_ms": [199.0] * n,
                "bada_gamma_rad": [0.009] * n,
            }
        )
        bada_data.write_parquet(bada_dir / "flight001.parquet")

        return config

    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm_pipeline.commands.visualize._require_viz")
    @patch("node_fdm_bada.utils.tas_to_cas")
    @patch("node_fdm_bada.utils.cas_to_mach")
    def test_visualize_creates_file(
        self,
        mock_cas_to_mach: MagicMock,
        mock_tas_to_cas: MagicMock,
        _mock_require_viz: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """PDF file created at expected path."""
        import types

        config = self._make_config_and_data(tmp_path)
        n = 20

        # Build a Delta Table DataFrame for the flight
        mock_read_delta.return_value = pl.DataFrame(
            {
                "meta_flight_id": ["flight001"] * n,
                "meta_aircraft_type": ["A320"] * n,
                "meta_split": ["test"] * n,
                "fdm_flag_valid": [True] * n,
                "raw_alt_m": [10000.0] * n,
                "era_tas_ms": [200.0] * n,
                "fdm_gamma_rad": [0.01] * n,
                "temp_k": [220.0] * n,
                "fdm_mcp_alt_sel_m": [10000.0] * n,
            }
        )

        # Mock conversion functions
        mock_tas_to_cas.return_value = np.zeros(n)
        mock_cas_to_mach.return_value = np.zeros(n)

        # Build mock matplotlib.pyplot as a real module type
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        axes_array = np.empty(3, dtype=object)
        axes_array[:] = [mock_ax, mock_ax, mock_ax]

        mock_plt = types.ModuleType("matplotlib.pyplot")
        mock_plt.subplots = MagicMock(return_value=(mock_fig, axes_array))  # type: ignore[attr-defined]
        mock_plt.close = MagicMock()  # type: ignore[attr-defined]
        mock_plt.tight_layout = MagicMock()  # type: ignore[attr-defined]

        mock_mpl = types.ModuleType("matplotlib")

        # Remove any cached matplotlib modules so our fake gets picked up
        saved = {}
        for key in list(sys.modules):
            if key.startswith("matplotlib"):
                saved[key] = sys.modules.pop(key)

        sys.modules["matplotlib"] = mock_mpl
        sys.modules["matplotlib.pyplot"] = mock_plt
        try:
            run_visualize(arch="opensky", config=config, typecode="A320")
        finally:
            # Restore original state
            for key in list(sys.modules):
                if key.startswith("matplotlib"):
                    del sys.modules[key]
            sys.modules.update(saved)

        # Figure saved
        mock_fig.savefig.assert_called_once()
        saved_path = mock_fig.savefig.call_args[0][0]
        assert "viz_A320_flight001.pdf" in str(saved_path)


class TestRunPlotPerformance:
    """Tests for ``run_plot_performance``."""

    def test_plot_performance_requires_file(self, tmp_path: Path) -> None:
        """SystemExit when performance.parquet is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        with pytest.raises(SystemExit, match="fdm evaluate"):
            run_plot_performance(config=config)


class TestRunPlotExample:
    """Tests for ``run_plot_example``."""

    def test_plot_example_requires_file(self, tmp_path: Path) -> None:
        """SystemExit when example.parquet is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        with pytest.raises(SystemExit, match="fdm visualize"):
            run_plot_example(config=config)
