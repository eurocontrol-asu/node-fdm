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


def _make_config(tmp_path: Path) -> Path:
    """Create a minimal pipeline config YAML."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
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


def _make_performance_parquet(data_dir: Path, *, n_rows: int = 6) -> Path:
    """Create a minimal performance.parquet with required columns."""
    df = pl.DataFrame(
        {
            "Aircraft": ["A320"] * n_rows,
            "Variable": [
                "Altitude [m]",
                "True airspeed [m/s]",
                "Flight path angle [deg]",
            ]
            * (n_rows // 3),
            "Model": ["PRED", "BADA"] * (n_rows // 2),
            "Phase": ["climb"] * n_rows,
            "MAE": [100.0] * n_rows,
        }
    )
    path = data_dir / "performance.parquet"
    df.write_parquet(path)
    return path


def _make_example_parquet(data_dir: Path, *, n_rows: int = 10) -> Path:
    """Create a minimal example.parquet with required columns."""
    df = pl.DataFrame(
        {
            "timestamp": list(range(n_rows)),
            "alt_std_m": [10000.0] * n_rows,
            "bada_alt_std_m": [10020.0] * n_rows,
            "pred_alt_std_m": [10010.0] * n_rows,
            "alt_sel_m": [10000.0] * n_rows,
            "cas_ms": [200.0] * n_rows,
            "bada_cas_ms": [199.0] * n_rows,
            "pred_cas_ms": [201.0] * n_rows,
            "cas_sel_ms": [200.0] * n_rows,
            "vz_ms": [5.0] * n_rows,
            "bada_vz_ms": [4.8] * n_rows,
            "pred_vz_ms": [5.1] * n_rows,
            "vz_sel_ms": [5.0] * n_rows,
        }
    )
    path = data_dir / "example.parquet"
    df.write_parquet(path)
    return path


class TestRunPlotPerformance:
    """Tests for ``run_plot_performance``."""

    def test_plot_performance_requires_file(self, tmp_path: Path) -> None:
        """SystemExit when performance.parquet is missing."""
        config = _make_config(tmp_path)

        with pytest.raises(SystemExit, match="fdm evaluate"):
            run_plot_performance(config=config)

    @patch("node_fdm_pipeline.commands.visualize._require_viz")
    def test_plot_performance_mocked(self, _mock_viz: MagicMock, tmp_path: Path) -> None:
        """Altair chart generated and saved for each aircraft."""
        import types

        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"
        _make_performance_parquet(data_dir)

        mock_hconcat = MagicMock()
        mock_hconcat.properties.return_value = mock_hconcat
        mock_hconcat.configure_title.return_value = mock_hconcat
        mock_hconcat.configure_axisX.return_value = mock_hconcat
        mock_hconcat.configure_facet.return_value = mock_hconcat

        mock_alt = types.ModuleType("altair")
        mock_base = MagicMock()
        mock_alt.Chart = MagicMock(  # type: ignore[attr-defined]
            return_value=MagicMock(
                mark_bar=MagicMock(
                    return_value=MagicMock(
                        encode=MagicMock(
                            return_value=MagicMock(properties=MagicMock(return_value=mock_base))
                        )
                    )
                )
            )
        )
        mock_base.transform_filter.return_value.encode.return_value = mock_base
        mock_alt.hconcat = MagicMock(return_value=mock_hconcat)  # type: ignore[attr-defined]
        mock_alt.X = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Y = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Row = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Color = MagicMock()  # type: ignore[attr-defined]

        saved_alt = sys.modules.get("altair")
        sys.modules["altair"] = mock_alt
        try:
            run_plot_performance(config=config)
        finally:
            if saved_alt is not None:
                sys.modules["altair"] = saved_alt
            else:
                sys.modules.pop("altair", None)

        mock_hconcat.save.assert_called_once()
        saved_path = mock_hconcat.save.call_args[0][0]
        assert "performance_A320.pdf" in str(saved_path)

    @patch("node_fdm_pipeline.commands.visualize._require_viz")
    def test_plot_performance_empty_parquet(self, _mock_viz: MagicMock, tmp_path: Path) -> None:
        """Empty performance.parquet generates no charts without crash."""
        import types

        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"
        empty_df = pl.DataFrame(
            {
                "Aircraft": pl.Series([], dtype=pl.Utf8),
                "Variable": pl.Series([], dtype=pl.Utf8),
                "Model": pl.Series([], dtype=pl.Utf8),
                "Phase": pl.Series([], dtype=pl.Utf8),
                "MAE": pl.Series([], dtype=pl.Float64),
            }
        )
        empty_df.write_parquet(data_dir / "performance.parquet")

        mock_alt = types.ModuleType("altair")
        mock_alt.Chart = MagicMock()  # type: ignore[attr-defined]
        mock_alt.X = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Y = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Row = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Color = MagicMock()  # type: ignore[attr-defined]

        saved_alt = sys.modules.get("altair")
        sys.modules["altair"] = mock_alt
        try:
            # Should not crash — empty unique list = no chart iterations
            run_plot_performance(config=config)
        finally:
            if saved_alt is not None:
                sys.modules["altair"] = saved_alt
            else:
                sys.modules.pop("altair", None)


class TestRunPlotExample:
    """Tests for ``run_plot_example``."""

    def test_plot_example_requires_file(self, tmp_path: Path) -> None:
        """SystemExit when example.parquet is missing."""
        config = _make_config(tmp_path)

        with pytest.raises(SystemExit, match="fdm visualize"):
            run_plot_example(config=config)

    @patch("node_fdm_pipeline.commands.visualize._require_viz")
    def test_plot_example_mocked(self, _mock_viz: MagicMock, tmp_path: Path) -> None:
        """Altair vconcat chart generated and saved."""
        import types

        config = _make_config(tmp_path)
        data_dir = tmp_path / "data"
        _make_example_parquet(data_dir)

        mock_vconcat = MagicMock()
        mock_vconcat.configure_axis.return_value = mock_vconcat

        mock_alt = types.ModuleType("altair")
        mock_base = MagicMock()
        mock_alt.Chart = MagicMock(  # type: ignore[attr-defined]
            return_value=MagicMock(
                mark_line=MagicMock(
                    return_value=MagicMock(
                        encode=MagicMock(
                            return_value=MagicMock(properties=MagicMock(return_value=mock_base))
                        )
                    )
                )
            )
        )
        chain = mock_base.transform_fold.return_value.transform_calculate.return_value
        chain.transform_calculate.return_value.encode.return_value = mock_base
        mock_alt.vconcat = MagicMock(return_value=mock_vconcat)  # type: ignore[attr-defined]
        mock_alt.X = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Y = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Color = MagicMock()  # type: ignore[attr-defined]
        mock_alt.StrokeDash = MagicMock()  # type: ignore[attr-defined]
        mock_alt.Scale = MagicMock()  # type: ignore[attr-defined]

        saved_alt = sys.modules.get("altair")
        sys.modules["altair"] = mock_alt
        try:
            run_plot_example(config=config)
        finally:
            if saved_alt is not None:
                sys.modules["altair"] = saved_alt
            else:
                sys.modules.pop("altair", None)

        mock_vconcat.save.assert_called_once()
        saved_path = mock_vconcat.save.call_args[0][0]
        assert "traj_example.pdf" in str(saved_path)
