"""Tests for visualization commands."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        """Create config, dirs, and test flight data."""
        data_dir = tmp_path / "data"
        process_dir = data_dir / "processed_flights"
        predict_dir = data_dir / "predicted_flights" / "A320"
        bada_dir = data_dir / "bada_flights" / "A320"
        figure_dir = data_dir / "figures"
        process_dir.mkdir(parents=True)
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

        # Create dummy flight data
        n = 20
        flight_data = pl.DataFrame(
            {
                "alt_std_m": [10000.0] * n,
                "tas_ms": [200.0] * n,
                "gamma_rad": [0.01] * n,
                "temp_k": [220.0] * n,
                "alt_sel_m": [10000.0] * n,
            }
        )
        flight_path = process_dir / "A320" / "flight001.parquet"
        flight_path.parent.mkdir(parents=True, exist_ok=True)
        flight_data.write_parquet(flight_path)

        # Create pred and bada parquet
        pred_data = pl.DataFrame(
            {
                "pred_alt_std_m": [10050.0] * n,
                "pred_tas_ms": [201.0] * n,
                "pred_gamma_rad": [0.011] * n,
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

        # Create split CSV
        split_df = pl.DataFrame(
            {
                "filepath": [str(flight_path)],
                "icao": ["abc123"],
                "split": ["test"],
                "aircraft_type": ["A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        return config

    @patch("node_fdm_pipeline.commands.visualize._require_viz")
    @patch("node_fdm_bada.utils.tas_to_cas")
    @patch("node_fdm_bada.utils.cas_to_mach")
    @patch("node_fdm_data.processor.FlightProcessor")
    def test_visualize_creates_file(
        self,
        mock_processor_cls: MagicMock,
        mock_cas_to_mach: MagicMock,
        mock_tas_to_cas: MagicMock,
        mock_require_viz: MagicMock,
        tmp_path: Path,
    ) -> None:
        """PDF file created at expected path."""
        import types

        import numpy as np

        config = self._make_config_and_data(tmp_path)

        # Mock processor
        mock_proc = MagicMock()
        mock_proc.process.return_value.collect.return_value = None
        mock_processor_cls.return_value = mock_proc

        # Mock conversion functions
        mock_tas_to_cas.return_value = np.zeros(20)
        mock_cas_to_mach.return_value = np.zeros(20)

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
