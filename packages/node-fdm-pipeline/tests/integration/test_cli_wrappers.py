"""Integration tests for CLI wrappers (in-process invocation)."""

from __future__ import annotations

from pathlib import Path


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
        """CLI table-info wrapper delegates to node_fdm_data.delta.table_info."""
        from unittest.mock import patch

        from node_fdm_pipeline.cli import table_info_cmd

        mock_info = {"partitions": ["20250101"], "columns": ["raw_icao24"], "versions": 1}
        with patch(
            "node_fdm_data.delta.table_info",
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
