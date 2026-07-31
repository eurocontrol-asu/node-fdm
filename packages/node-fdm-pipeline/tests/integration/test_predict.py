"""Tests for prediction commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from node_fdm_pipeline.commands.predict import run_predict, run_predict_bada

if TYPE_CHECKING:
    from node_fdm_pipeline.resolver import ArchitectureInfo


def _make_delta_df(
    *,
    n: int = 10,
    acft: str = "A320",
    flight_id: str = "F001",
    split: str = "test",
    extra_cols: dict[str, list[object]] | None = None,
) -> pl.DataFrame:
    """Build a minimal Delta Table DataFrame for predict tests."""
    data: dict[str, list[object]] = {
        "meta_flight_id": [flight_id] * n,
        "meta_aircraft_type": [acft] * n,
        "meta_split": [split] * n,
        "fdm_flag_valid": [True] * n,
        "fdm_flag_crop_start": [0] * n,
        "fdm_flag_crop_end": [n - 1] * n,
        "raw_timestamp": list(range(n)),
        # columns consumed by run_predict_bada → _run_predict_bada_typecode
        "raw_alt_m": [3000.0] * n,
        "era_tas_ms": [230.0] * n,
        "era_temp_K": [288.15] * n,
        "era_mach": [0.82] * n,
        "fdm_cas_ms": [144.0] * n,
        "fdm_cas_sel_ms": [144.0] * n,
        "fdm_mach_sel": [0.82] * n,
        "fdm_vz_sel_ms": [0.0] * n,
        "fdm_alt_target_m": [10668.0] * n,
        "fdm_long_wind_ms": [5.0] * n,
        # common feature columns used by mocked architectures
        "distance_m": [float(i * 1000) for i in range(n)],
        "alt_sel_m": [10668.0] * n,
        "long_wind_ms": [5.0] * n,
        "gs_ms": [230.0] * n,
        # NaN-filter test columns
        "altitude_ft": [35000.0] * n,
        "mach_sel": [0.82] * n,
        "cas_sel_kt": [280.0] * n,
        "groundspeed": [450.0] * n,
        "vertical_rate": [0.0] * n,
    }
    if extra_cols:
        data.update(extra_cols)
    return pl.DataFrame(data)


class TestRunPredict:
    """Tests for ``run_predict``."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create YAML config and model directory."""
        data_dir = tmp_path / "data"
        models_dir = data_dir / "models"
        models_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        model_dir = models_dir / "node_adsb_v1_A320"
        model_dir.mkdir()

        return config

    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_output_format(
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Output parquet has pred_* columns."""
        config = self._make_config(tmp_path)

        mock_read_delta.return_value = _make_delta_df()

        from node_fdm_pipeline.resolver import ArchitectureInfo

        mock_resolve.return_value = ArchitectureInfo(
            name="node_adsb_v1",
            x_cols=["distance_m"],
            u_cols=["alt_sel_m"],
            e0_cols=["long_wind_ms"],
            dx_cols=[(1, "gs_ms")],
            preprocessing_fn=None,
            segment_filter_fn=None,
        )

        mock_predictor = MagicMock()
        mock_predictor.predict_flight.return_value = {
            "alt_std_m": np.zeros(10),
            "tas_ms": np.ones(10),
        }
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="adsb",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # Check output dir was created
        predict_dir = tmp_path / "data" / "predicted_flights" / "node_adsb_v1_A320" / "A320"
        assert predict_dir.exists()

        # Check predictor was called
        mock_predictor.predict_flight.assert_called_once()

    @patch("node_fdm_data.delta.read_delta_table")
    def test_predict_missing_model(
        self,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Warning logged when model doesn't exist, typecode skipped."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        mock_read_delta.return_value = _make_delta_df()

        # Should not raise — just logs warning and skips
        run_predict(
            arch="adsb",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm.predictor.NodeFDMPredictor")
    def test_predict_empty_test_set(
        self,
        mock_predictor_cls: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Warning logged when no test flights for typecode."""
        config = self._make_config(tmp_path)

        # Return a DataFrame with no matching A320 test rows
        mock_read_delta.return_value = _make_delta_df(split="train")

        # Should not raise — just logs warning and skips
        run_predict(
            arch="adsb",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # Predictor should not have been used for prediction
        mock_predictor_cls.return_value.predict_flight.assert_not_called()


class TestRunPredictBada:
    """Tests for ``run_predict_bada``."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create config with BADA settings."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
  - UNKNOWN

bada:
  bada_4_2_dir: "{tmp_path / "bada"}"
"""
        )

        return config

    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm_bada.aircraft_mapping.get_bada_identifier")
    def test_predict_bada_missing_mapping(
        self,
        mock_get_bada: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Unknown typecode → warning logged, skipped, continues."""
        config = self._make_config(tmp_path)

        mock_read_delta.return_value = _make_delta_df(acft="UNKNOWN")
        mock_get_bada.side_effect = KeyError("UNKNOWN not found")

        # Should not raise — graceful skip
        run_predict_bada(config=config, typecode="UNKNOWN", jobs=1)

        mock_get_bada.assert_called_once_with("UNKNOWN")

    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm_bada.aircraft_mapping.get_bada_identifier")
    def test_predict_bada_success(
        self,
        mock_get_bada: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Valid typecode → Parallel invoked, output dir created."""
        import sys

        config = self._make_config(tmp_path)

        mock_read_delta.return_value = _make_delta_df()
        mock_get_bada.return_value = "A320_BADA"

        # Inject fake pyBADA module into sys.modules
        mock_bada4 = MagicMock()
        mock_bada_pkg = MagicMock()
        mock_bada_pkg.bada4 = mock_bada4
        mock_parallel = MagicMock(return_value=MagicMock(return_value=[]))

        with (
            patch.dict(sys.modules, {"pyBADA": mock_bada_pkg, "pyBADA.bada4": mock_bada4}),
            patch.dict(
                sys.modules, {"joblib": MagicMock(Parallel=mock_parallel, delayed=lambda f: f)}
            ),
        ):
            run_predict_bada(config=config, typecode="A320", jobs=2)

        mock_get_bada.assert_called_once_with("A320")
        mock_parallel.assert_called_once()

        # Output dir created
        bada_dir = tmp_path / "data" / "bada_flights" / "A320"
        assert bada_dir.exists()

    @patch("node_fdm_data.delta.read_delta_table")
    def test_predict_bada_missing_split(
        self,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No test flights → warning logged, no crash."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )
        # No test flights → empty after filter
        mock_read_delta.return_value = _make_delta_df(split="train")

        # Should not raise — just logs warning and skips
        run_predict_bada(config=config, typecode="A320", jobs=1)

    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm_bada.aircraft_mapping.get_bada_identifier")
    def test_predict_bada_empty_test_set(
        self,
        mock_get_bada: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Empty test set → warning logged, no crash."""
        import sys

        config = self._make_config(tmp_path)

        # A320 has train only
        mock_read_delta.return_value = _make_delta_df(split="train")

        mock_get_bada.return_value = "A320_BADA"
        mock_bada4 = MagicMock()
        mock_bada_pkg = MagicMock()
        mock_bada_pkg.bada4 = mock_bada4

        with patch.dict(sys.modules, {"pyBADA": mock_bada_pkg, "pyBADA.bada4": mock_bada4}):
            run_predict_bada(config=config, typecode="A320", jobs=1)


class TestPredictNanFiltering:
    """Regression tests for AXM-493: filter NaN segments in prediction."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create config and model directory."""
        data_dir = tmp_path / "data"
        models_dir = data_dir / "models"
        models_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        model_dir = models_dir / "node_adsb_v1_A320"
        model_dir.mkdir()

        return config

    def _make_nan_delta_df(self, *, nan_fraction: float, n: int = 100) -> pl.DataFrame:
        """Build a Delta Table DataFrame with controllable NaN fraction in mach_sel."""
        nan_count = int(n * nan_fraction)
        mach_sel = [float("nan")] * nan_count + [0.82] * (n - nan_count)
        return pl.DataFrame(
            {
                "meta_flight_id": ["F001"] * n,
                "meta_aircraft_type": ["A320"] * n,
                "meta_split": ["test"] * n,
                "fdm_flag_valid": [True] * n,
                "fdm_flag_crop_start": [0] * n,
                "fdm_flag_crop_end": [n - 1] * n,
                "raw_timestamp": list(range(n)),
                "altitude_ft": [35000.0] * n,
                "mach_sel": mach_sel,
                "cas_sel_kt": [280.0] * n,
                "groundspeed": [450.0] * n,
                "vertical_rate": [0.0] * n,
            }
        )

    def _mock_architecture(self) -> ArchitectureInfo:
        """Build a mock ArchitectureInfo that passes through columns."""
        from node_fdm_pipeline.resolver import ArchitectureInfo

        def identity_preprocess(df: pl.DataFrame) -> pl.DataFrame:
            return df

        return ArchitectureInfo(
            name="node_adsb_v1",
            x_cols=["altitude_ft"],
            u_cols=["mach_sel", "cas_sel_kt"],
            e0_cols=["groundspeed"],
            dx_cols=[(1, "vertical_rate")],
            preprocessing_fn=identity_preprocess,
            segment_filter_fn=None,
        )

    @pytest.mark.parametrize(
        "nan_fraction, expected_rows",
        [
            pytest.param(0.0, 100, id="clean_flight_all_rows"),
            # Post-AXM (commit 53025e5): _load_test_df ffill/bfills numeric
            # state/exo columns and predict no longer drops NaN rows --- the
            # full timeline is forwarded to the predictor (crop_start/end is
            # the only slicing knob). Non-prefixed columns like ``mach_sel``
            # remain NaN but the row count is preserved.
            pytest.param(0.6, 100, id="nan_rows_passthrough"),
            pytest.param(0.9, None, id="skip_above_threshold"),
        ],
    )
    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_handles_nan_fraction(  # noqa: PLR0913
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
        nan_fraction: float,
        expected_rows: int | None,
    ) -> None:
        """Predict either runs on finite rows or skips a flight depending on NaN fraction."""
        config = self._make_config(tmp_path)
        mock_read_delta.return_value = self._make_nan_delta_df(nan_fraction=nan_fraction)
        mock_resolve.return_value = self._mock_architecture()

        mock_predictor = MagicMock()
        mock_predictor.predict_flight.side_effect = lambda _x, u_seq, _e: {
            "altitude_ft": np.zeros(len(u_seq))
        }
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="adsb",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        if expected_rows is None:
            mock_predictor.predict_flight.assert_not_called()
            output_file = tmp_path / "data" / "predicted_flights" / "A320" / "F001.parquet"
            assert not output_file.exists()
        else:
            mock_predictor.predict_flight.assert_called_once()
            u_seq_arg = mock_predictor.predict_flight.call_args[0][1]
            assert len(u_seq_arg) == expected_rows


class TestPredictMissingSplit:
    """Edge case: no test flights for run_predict."""

    @patch("node_fdm_data.delta.read_delta_table")
    def test_predict_missing_split_csv(
        self,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No test flights and no model → logs warning, no crash."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )
        # Return train-only data → no test flights
        mock_read_delta.return_value = _make_delta_df(split="train")

        # Should not raise — no model found, skips gracefully
        run_predict(arch="adsb", config=config, typecode="A320", local_model=True)


class TestPredictXInitGuard:
    """Regression tests for AXM-494: x_init finite guard in predict_flight."""

    @patch("node_fdm_data.delta.read_delta_table")
    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_skips_flight_on_bad_x_init(
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        mock_read_delta: MagicMock,
        tmp_path: Path,
    ) -> None:
        """predict_flight raising ValueError → flight skipped, no output."""
        from node_fdm_pipeline.resolver import ArchitectureInfo

        data_dir = tmp_path / "data"
        models_dir = data_dir / "models"
        models_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        model_dir = models_dir / "node_adsb_v1_A320"
        model_dir.mkdir()

        mock_read_delta.return_value = _make_delta_df()

        mock_preprocessing = MagicMock()
        mock_result = MagicMock()
        mock_result.select.return_value.to_numpy.return_value = np.zeros((10, 1), dtype=np.float32)
        mock_preprocessing.return_value = mock_result

        mock_resolve.return_value = ArchitectureInfo(
            name="node_adsb_v1",
            x_cols=["distance_m"],
            u_cols=["alt_sel_m"],
            e0_cols=["long_wind_ms"],
            dx_cols=[(1, "gs_ms")],
            preprocessing_fn=mock_preprocessing,
            segment_filter_fn=None,
        )

        mock_predictor = MagicMock()
        mock_predictor.predict_flight.side_effect = ValueError(
            "x_init contains non-finite values: altitude_m=nan"
        )
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="adsb",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        mock_predictor.predict_flight.assert_called_once()

        # No output parquet written for the skipped flight
        output_file = tmp_path / "data" / "predicted_flights" / "A320" / "F001.parquet"
        assert not output_file.exists()
