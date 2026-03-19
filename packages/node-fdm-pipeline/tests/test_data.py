"""Tests for data pipeline commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import polars as pl
import pytest

from node_fdm_pipeline.commands.data import (
    _require_traffic,
    _split_at_gaps,
    aircraft_list,
    download,
    identify,
    process,
)

if TYPE_CHECKING:
    from traffic.core import Flight


class TestRequireTraffic:
    """Tests for the traffic import guard."""

    def test_require_traffic_missing(self) -> None:
        """Raises SystemExit when traffic is not installed."""
        with (
            patch.dict("sys.modules", {"traffic": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'traffic'")),
            pytest.raises(SystemExit, match="1"),
        ):
            _require_traffic()


class TestProcessCommand:
    """Tests for the ``process`` command (no traffic dependency)."""

    def test_process_dry_run(self, tmp_config: Path) -> None:
        """--dry-run validates config without writing files."""
        process(arch="opensky", config=tmp_config, dry_run=True)
        # No files should be created beyond the config dir

    def test_process_empty_preprocess_dir(self, tmp_path: Path, mocker: Any) -> None:
        """Warning log and clean return when no parquet files found."""
        # Create a config pointing at an empty preprocess dir
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"

typecodes:
  - A320
"""
        )

        mock_arco_cls = mocker.MagicMock()
        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        process(arch="opensky", config=config, dry_run=False)

        # No output files — clean return on empty dir
        assert not process_dir.exists() or len(list(process_dir.iterdir())) == 0

    def test_process_single_file(self, tmp_path: Path, mocker: Any) -> None:
        """Process a synthetic parquet and verify output exists."""
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)
        era5_cache = data_dir / "era5_cache"
        era5_cache.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"
  era5_cache_dir: "era5_cache"

typecodes:
  - A320
"""
        )

        n = 50
        # Create a synthetic parquet with all required columns
        df = pl.DataFrame(
            {
                "flight_id": ["F001"] * n,
                "timestamp": [float(i * 4) for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "selected_mcp": [35000.0] * n,
                "vertical_rate": [100.0] * n,
                "Mach": [0.78] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                "latitude": [48.0 + i * 0.001 for i in range(n)],
                "longitude": [2.0 + i * 0.001 for i in range(n)],
                "track": [90.0] * n,
                "heading": [88.0] * n,
                "typecode": ["A320"] * n,
                "icao24": ["abc123"] * n,
                "adep_dist": [100.0 - i for i in range(n)],
                "ades_dist": [float(i * 2) for i in range(n)],
            }
        )
        df.write_parquet(preprocess_dir / "processed_20250101.parquet")

        # Mock fastmeteo — interpolate adds weather columns
        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            pdf["temperature"] = 220.0
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        mock_arco_cls = mocker.MagicMock()
        mock_arco_instance = mocker.MagicMock()
        mock_arco_instance.interpolate.side_effect = fake_interpolate
        mock_arco_cls.return_value = mock_arco_instance

        # Patch the full import chain: fastmeteo.source.arco_era5.ArcoEra5
        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        process(arch="opensky", config=config, dry_run=False)

        output = process_dir / "processed_20250101.parquet"
        assert output.exists()

        # Verify output has the new derived columns
        result = pl.read_parquet(output)
        assert "fdm_gamma_rad" in result.columns
        assert "fdm_long_wind_kt" in result.columns
        assert "fdm_mach_sel" in result.columns
        assert "distance_along_track_m" in result.columns
        # Verify lateral augmentation columns
        assert "track_ortho" in result.columns
        assert "drift_angle" in result.columns

    def test_process_skip_existing(self, tmp_path: Path, mocker: Any) -> None:
        """Already-processed files are skipped."""
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"

typecodes:
  - A320
"""
        )

        mock_arco_cls = mocker.MagicMock()
        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        # Create input AND output so it should skip
        df = pl.DataFrame({"x": [1, 2, 3]})
        df.write_parquet(preprocess_dir / "file.parquet")
        df_out = pl.DataFrame({"flight_id": ["F1", "F2"], "typecode": ["A320", "A320"]})
        df_out.write_parquet(process_dir / "file.parquet")

        # Should not raise — skip with info log
        process(arch="opensky", config=config, dry_run=False)

    # --- ERA5 post-interpolation validation (AXM-491) ---

    def _make_process_env(
        self,
        tmp_path: Path,
        mocker: Any,
        *,
        interpolate_fn: Any,
    ) -> tuple[Path, Path]:
        """Set up dirs, config, synthetic parquet, and fastmeteo mock.

        Returns ``(config_path, output_path)`` for assertion.
        """
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)
        era5_cache = data_dir / "era5_cache"
        era5_cache.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"
  era5_cache_dir: "era5_cache"

typecodes:
  - A320
"""
        )

        n = 50
        df = pl.DataFrame(
            {
                "flight_id": ["F001"] * n,
                "timestamp": [float(i * 4) for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "selected_mcp": [35000.0] * n,
                "vertical_rate": [100.0] * n,
                "Mach": [0.78] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                "latitude": [48.0 + i * 0.001 for i in range(n)],
                "longitude": [2.0 + i * 0.001 for i in range(n)],
                "track": [90.0] * n,
                "heading": [88.0] * n,
                "typecode": ["A320"] * n,
                "icao24": ["abc123"] * n,
                "adep_dist": [100.0 - i for i in range(n)],
                "ades_dist": [float(i * 2) for i in range(n)],
            }
        )
        df.write_parquet(preprocess_dir / "processed_20250101.parquet")

        mock_arco_cls = mocker.MagicMock()
        mock_arco_instance = mocker.MagicMock()
        mock_arco_instance.interpolate.side_effect = interpolate_fn
        mock_arco_cls.return_value = mock_arco_instance

        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        output = process_dir / "processed_20250101.parquet"
        return config, output

    def test_process_detects_missing_era5_columns(self, tmp_path: Path, mocker: Any) -> None:
        """Interpolate returns df without ERA5 cols → file skipped."""

        def fake_interpolate(pdf: Any) -> Any:
            return pdf.copy()  # no ERA5 cols added

        config, output = self._make_process_env(tmp_path, mocker, interpolate_fn=fake_interpolate)
        process(arch="opensky", config=config, dry_run=False)

        assert not output.exists()

    def test_process_detects_all_null_era5(self, tmp_path: Path, mocker: Any) -> None:
        """Interpolate returns df with all-null ERA5 cols → file skipped."""
        import numpy as np

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            pdf["temperature"] = np.nan
            pdf["u_component_of_wind"] = np.nan
            pdf["v_component_of_wind"] = np.nan
            return pdf

        config, output = self._make_process_env(tmp_path, mocker, interpolate_fn=fake_interpolate)
        process(arch="opensky", config=config, dry_run=False)

        assert not output.exists()

    def test_process_era5_partial_null(self, tmp_path: Path, mocker: Any) -> None:
        """ERA5 with 10% null temperature → file skipped (> 5% threshold)."""
        import numpy as np

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            n = len(pdf)
            temps = [220.0] * n
            # Set 10% of rows to null (via NaN in pandas → null in Polars)
            for i in range(n // 10):
                temps[i] = np.nan
            pdf["temperature"] = temps
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        config, output = self._make_process_env(tmp_path, mocker, interpolate_fn=fake_interpolate)
        process(arch="opensky", config=config, dry_run=False)

        assert not output.exists()

    def test_process_era5_below_threshold(self, tmp_path: Path, mocker: Any) -> None:
        """ERA5 with 2% null temperature → file processed, null rows dropped."""
        import numpy as np

        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)
        era5_cache = data_dir / "era5_cache"
        era5_cache.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"
  era5_cache_dir: "era5_cache"

typecodes:
  - A320
"""
        )

        n = 51
        df = pl.DataFrame(
            {
                "flight_id": ["F001"] * n,
                "timestamp": [float(i * 4) for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "selected_mcp": [35000.0] * n,
                "vertical_rate": [100.0] * n,
                "Mach": [0.78] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                # Small lat/lon step so dropping any row won't create
                # a > 200 m gap in cumulative_distance after unique() reorder
                "latitude": [48.0 + i * 0.0001 for i in range(n)],
                "longitude": [2.0 + i * 0.0001 for i in range(n)],
                "track": [90.0] * n,
                "heading": [88.0] * n,
                "typecode": ["A320"] * n,
                "icao24": ["abc123"] * n,
                "adep_dist": [100.0 - i for i in range(n)],
                "ades_dist": [float(i * 2) for i in range(n)],
            }
        )
        df.write_parquet(preprocess_dir / "processed_20250101.parquet")

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            temps = [220.0] * len(pdf)
            # Set last row to NaN (< 5% threshold for 55 rows)
            temps[-1] = np.nan
            pdf["temperature"] = temps
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        mock_arco_cls = mocker.MagicMock()
        mock_arco_instance = mocker.MagicMock()
        mock_arco_instance.interpolate.side_effect = fake_interpolate
        mock_arco_cls.return_value = mock_arco_instance

        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        output = process_dir / "processed_20250101.parquet"
        process(arch="opensky", config=config, dry_run=False)

        assert output.exists()
        result = pl.read_parquet(output)
        # The null row should have been dropped
        assert result["temperature"].null_count() == 0
        assert result["temperature"].is_nan().sum() == 0

    def test_process_era5_nan_values(self, tmp_path: Path, mocker: Any) -> None:
        """ERA5 with NaN (not null) temperature above threshold → file skipped."""

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            n = len(pdf)
            temps = [220.0] * n
            # Set 10% of rows to NaN (float, not pandas NA)
            for i in range(n // 10):
                temps[i] = float("nan")
            pdf["temperature"] = temps
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        config, output = self._make_process_env(tmp_path, mocker, interpolate_fn=fake_interpolate)
        process(arch="opensky", config=config, dry_run=False)

        assert not output.exists()

    def test_process_era5_one_column_above_threshold(self, tmp_path: Path, mocker: Any) -> None:
        """One ERA5 column at 6% null, others at 0% → file skipped entirely."""
        import numpy as np

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            n = len(pdf)
            temps = [220.0] * n
            # Set 3 rows to null (6% of 50 rows)
            for i in range(3):
                temps[i] = np.nan
            pdf["temperature"] = temps
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        config, output = self._make_process_env(tmp_path, mocker, interpolate_fn=fake_interpolate)
        process(arch="opensky", config=config, dry_run=False)

        assert not output.exists()

    def test_process_era5_happy_path(self, tmp_path: Path, mocker: Any) -> None:
        """Interpolate returns df with valid ERA5 cols → file processed."""

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            pdf["temperature"] = 220.0
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        config, output = self._make_process_env(tmp_path, mocker, interpolate_fn=fake_interpolate)
        process(arch="opensky", config=config, dry_run=False)

        assert output.exists()
        result = pl.read_parquet(output)
        assert "temperature" in result.columns

    def test_process_drops_null_coords(self, tmp_path: Path, mocker: Any) -> None:
        """Null lat/lon rows are dropped before ERA5 — no NaN in output (AXM-511)."""
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)
        era5_cache = data_dir / "era5_cache"
        era5_cache.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"
  era5_cache_dir: "era5_cache"

typecodes:
  - A320
"""
        )

        n = 55
        n_nulls = 5
        lats: list[float | None] = [48.0 + i * 0.001 for i in range(n)]
        lons: list[float | None] = [2.0 + i * 0.001 for i in range(n)]
        # Inject null coordinates at the tail (last n_nulls rows)
        for idx in range(n - n_nulls, n):
            lats[idx] = None
            lons[idx] = None

        df = pl.DataFrame(
            {
                "flight_id": ["F001"] * n,
                "timestamp": [float(i * 4) for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "selected_mcp": [35000.0] * n,
                "vertical_rate": [100.0] * n,
                "Mach": [0.78] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                "latitude": lats,
                "longitude": lons,
                "track": [90.0] * n,
                "heading": [88.0] * n,
                "typecode": ["A320"] * n,
                "icao24": ["abc123"] * n,
                "adep_dist": [100.0 - i for i in range(n)],
                "ades_dist": [float(i * 2) for i in range(n)],
            }
        )
        df.write_parquet(preprocess_dir / "processed_20250101.parquet")

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            pdf["temperature"] = 220.0
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        mock_arco_cls = mocker.MagicMock()
        mock_arco_instance = mocker.MagicMock()
        mock_arco_instance.interpolate.side_effect = fake_interpolate
        mock_arco_cls.return_value = mock_arco_instance

        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        process(arch="opensky", config=config, dry_run=False)

        output = process_dir / "processed_20250101.parquet"
        assert output.exists()

        result = pl.read_parquet(output)
        # Null-coord rows dropped; crop may trim further
        assert len(result) <= n - n_nulls
        assert len(result) > 0
        # No NaN in mach or distance columns
        assert result["era_mach"].is_nan().sum() == 0
        assert result["distance_along_track_m"].is_nan().sum() == 0

    def test_process_filters_mach_outliers(self, tmp_path: Path, mocker: Any) -> None:
        """Rows with Mach > 1.05 (ADS-B groundspeed outliers) are filtered out."""
        data_dir = tmp_path / "data"
        preprocess_dir = data_dir / "preprocess"
        preprocess_dir.mkdir(parents=True)
        process_dir = data_dir / "process"
        process_dir.mkdir(parents=True)
        era5_cache = data_dir / "era5_cache"
        era5_cache.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  preprocess_dir: "preprocess"
  process_dir: "process"
  era5_cache_dir: "era5_cache"

typecodes:
  - A320
"""
        )

        n_good = 48
        n_bad = 2
        n = n_good + n_bad

        # Good rows: TAS 450 kt → Mach ~0.78 at FL350/220K
        # Bad rows: TAS 630 kt → Mach ~1.09 (erroneous ADS-B groundspeed)
        tas_values = [450.0] * n_good + [630.0] * n_bad

        df = pl.DataFrame(
            {
                "flight_id": ["F001"] * n,
                "timestamp": [float(i * 4) for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "selected_mcp": [35000.0] * n,
                "vertical_rate": [100.0] * n,
                "Mach": [0.78] * n,
                "IAS": [280.0] * n,
                "TAS": tas_values,
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                "latitude": [48.0 + i * 0.001 for i in range(n)],
                "longitude": [2.0 + i * 0.001 for i in range(n)],
                "track": [90.0] * n,
                "heading": [88.0] * n,
                "typecode": ["A320"] * n,
                "icao24": ["abc123"] * n,
                "adep_dist": [100.0 - i for i in range(n)],
                "ades_dist": [float(i * 2) for i in range(n)],
            }
        )
        df.write_parquet(preprocess_dir / "processed_20250101.parquet")

        def fake_interpolate(pdf: Any) -> Any:
            pdf = pdf.copy()
            pdf["temperature"] = 220.0
            pdf["u_component_of_wind"] = 5.0
            pdf["v_component_of_wind"] = -3.0
            return pdf

        mock_arco_cls = mocker.MagicMock()
        mock_arco_instance = mocker.MagicMock()
        mock_arco_instance.interpolate.side_effect = fake_interpolate
        mock_arco_cls.return_value = mock_arco_instance

        mock_source_arco = mocker.MagicMock(ArcoEra5=mock_arco_cls)
        mock_source = mocker.MagicMock(arco_era5=mock_source_arco)
        mocker.patch.dict(
            "sys.modules",
            {
                "fastmeteo": mocker.MagicMock(),
                "fastmeteo.source": mock_source,
                "fastmeteo.source.arco_era5": mock_source_arco,
            },
        )

        process(arch="opensky", config=config, dry_run=False)

        output = process_dir / "processed_20250101.parquet"
        assert output.exists()

        result = pl.read_parquet(output)
        # All Mach values must be ≤ 1.05 — outlier rows filtered
        mach_max: float | None = result["era_mach"].cast(pl.Float64).max()  # type: ignore[assignment]
        assert mach_max is not None and mach_max <= 1.05
        assert len(result) > 0


class TestDownloadCommand:
    """Tests for the ``download`` command."""

    def test_download_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without writing files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "aircraft_db.csv").write_text(
            "icao24,registration,typecode,age,airline\nabc123,F-WXYZ,A320,5,AFR\n"
        )
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  download_dir: "download"

typecodes:
  - A320
"""
        )

        download(
            config=config,
            start_date="2025-01-01",
            end_date="2025-01-02",
            dry_run=True,
        )

        # No download dir created in dry-run
        download_dir = data_dir / "download"
        if download_dir.exists():
            assert len(list(download_dir.iterdir())) == 0

    def test_download_missing_aircraft_db(self, tmp_path: Path) -> None:
        """Raises SystemExit when aircraft_db.csv is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  download_dir: "download"

typecodes:
  - A320
"""
        )

        with pytest.raises(SystemExit):
            download(
                config=config,
                start_date="2025-01-01",
                end_date="2025-01-02",
                dry_run=False,
            )


class TestAircraftListCommand:
    """Tests for the ``aircraft-list`` command."""

    def test_aircraft_list_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without querying OpenSky."""
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{tmp_path / "data"}"

typecodes:
  - A320
"""
        )

        # dry_run returns before _require_traffic() is called
        aircraft_list(config=config, dry_run=True)


class TestCLINewCommands:
    """CLI smoke tests for new data commands."""

    def test_aircraft_list_help(self) -> None:
        """fdm aircraft-list --help shows expected flags."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "aircraft-list", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--config" in output
        assert "--dry-run" in output

    def test_download_help(self) -> None:
        """fdm download --help shows expected flags."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "download", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--start-date" in output
        assert "--end-date" in output
        assert "--dry-run" in output

    def test_preprocess_help(self) -> None:
        """fdm preprocess --help shows expected flags."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "preprocess", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--history-file" in output
        assert "--workers" in output
        assert "--dry-run" in output

    def test_process_help_has_dry_run(self) -> None:
        """fdm process --help now shows --dry-run flag."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "node_fdm_pipeline", "process", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0
        assert "--dry-run" in output


# ---------------------------------------------------------------------------
# Download → Delta Table tests (v3 étape 0)
# ---------------------------------------------------------------------------


class TestDownloadDelta:
    """Tests for download → Delta Table (v3 pipeline étape 0)."""

    @staticmethod
    def _make_raw_df(n: int = 10) -> pl.DataFrame:
        """Synthetic OpenSky data with raw + BDS columns (pre-rename)."""
        from datetime import datetime as dt

        return pl.DataFrame(
            {
                "timestamp": [dt(2025, 1, 1, 12, 0, i) for i in range(n)],
                "icao24": ["abc123"] * n,
                "callsign": ["TEST01"] * n,
                "latitude": [48.0 + i * 0.001 for i in range(n)],
                "longitude": [2.0 + i * 0.001 for i in range(n)],
                "altitude": [35000.0 + i * 10 for i in range(n)],
                "groundspeed": [440.0 + i * 0.1 for i in range(n)],
                "track": [90.0] * n,
                "vertical_rate": [100.0] * n,
                "selected_mcp": [35000.0] * n,
                "selected_fms": [35000.0] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "Mach": [0.78] * n,
                "heading": [88.0] * n,
            }
        )

    def test_download_creates_delta(self, tmp_path: Path) -> None:
        """Delta Table created with raw_* + bds_* columns."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        df = _rename_to_v3(self._make_raw_df(), batch_date="20250101")
        table_path = tmp_path / "flights.delta"
        write_columns(df, table_path)

        result = pl.read_delta(str(table_path))
        raw_cols = {c for c in result.columns if c.startswith("raw_")}
        bds_cols = {c for c in result.columns if c.startswith("bds_")}
        assert raw_cols == {
            "raw_timestamp",
            "raw_icao24",
            "raw_callsign",
            "raw_lat_deg",
            "raw_lon_deg",
            "raw_alt_ft",
            "raw_gs_kt",
            "raw_track_deg",
            "raw_vz_ftmin",
        }
        assert bds_cols == {
            "bds_mcp_sel_alt_ft",
            "bds_fms_sel_alt_ft",
            "bds_ias_kt",
            "bds_tas_kt",
            "bds_mach",
            "bds_hdg_deg",
        }
        assert "meta_batch_date" in result.columns

    def test_download_partition_key(self, tmp_path: Path) -> None:
        """2 dates → 2 partitions meta_batch_date."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        table_path = tmp_path / "flights.delta"
        df1 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250101")
        df2 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250102")
        combined = pl.concat([df1, df2])
        write_columns(combined, table_path)

        result = pl.read_delta(str(table_path))
        dates = result["meta_batch_date"].unique().sort().to_list()
        assert dates == ["20250101", "20250102"]

    def test_download_bds_preserved(self, tmp_path: Path) -> None:
        """BDS TAS and Mach preserved with correct column names."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        df = _rename_to_v3(self._make_raw_df(), batch_date="20250101")
        table_path = tmp_path / "flights.delta"
        write_columns(df, table_path)

        result = pl.read_delta(str(table_path))
        assert "bds_tas_kt" in result.columns
        assert "bds_mach" in result.columns
        assert result["bds_tas_kt"].to_list() == [450.0] * 10
        assert result["bds_mach"].to_list() == [0.78] * 10

    def test_download_idempotent(self, tmp_path: Path) -> None:
        """Download 2x same date -> partition overwritten, no duplicates."""
        from node_fdm_data.delta import write_columns

        from node_fdm_pipeline.commands.data import _rename_to_v3

        table_path = tmp_path / "flights.delta"

        # First write
        df1 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250101")
        write_columns(df1, table_path)

        # Second write (same date, same schema)
        df2 = _rename_to_v3(self._make_raw_df(n=5), batch_date="20250101")
        write_columns(df2, table_path)

        result = pl.read_delta(str(table_path))
        assert len(result) == 5  # overwritten, not appended


# ---------------------------------------------------------------------------
# identify tests (v3 étape 1)
# ---------------------------------------------------------------------------


def _make_delta_table(
    tmp_path: Path,
    *,
    icao24s: list[str],
    callsigns: list[str | None],
    timestamps_s: list[list[int]],
    batch_date: str = "20250101",
) -> Path:
    """Create a synthetic Delta Table with raw_* columns for identify tests.

    Each entry in icao24s/callsigns/timestamps_s defines one flight's points.
    """
    from datetime import UTC, timedelta
    from datetime import datetime as dt

    rows: dict[str, list[object]] = {
        "raw_timestamp": [],
        "raw_icao24": [],
        "raw_callsign": [],
        "raw_lat_deg": [],
        "raw_lon_deg": [],
        "raw_alt_ft": [],
        "raw_gs_kt": [],
        "raw_track_deg": [],
        "raw_vz_ftmin": [],
        "meta_batch_date": [],
    }
    base = dt(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    for icao24, callsign, ts_offsets in zip(icao24s, callsigns, timestamps_s, strict=False):
        for offset in ts_offsets:
            rows["raw_timestamp"].append(base + timedelta(seconds=offset))
            rows["raw_icao24"].append(icao24)
            rows["raw_callsign"].append(callsign)
            rows["raw_lat_deg"].append(48.0)
            rows["raw_lon_deg"].append(2.0)
            rows["raw_alt_ft"].append(35000.0)
            rows["raw_gs_kt"].append(440.0)
            rows["raw_track_deg"].append(90.0)
            rows["raw_vz_ftmin"].append(100.0)
            rows["meta_batch_date"].append(batch_date)

    df = pl.DataFrame(rows).cast({"raw_callsign": pl.Utf8})
    table_path = tmp_path / "flights.delta"
    from node_fdm_data.delta import write_columns

    write_columns(df, table_path)
    return table_path


def _make_flightlist(
    tmp_path: Path,
    *,
    entries: list[dict[str, str]],
    batch_date: str = "20250101",
) -> Path:
    """Create a synthetic flightlist parquet.

    Each entry: {icao24, callsign, departure, arrival, typecode, firstseen, lastseen}.
    """
    from datetime import UTC
    from datetime import datetime as dt

    rows: dict[str, list[object]] = {
        "icao24": [],
        "callsign": [],
        "departure": [],
        "arrival": [],
        "typecode": [],
        "firstseen": [],
        "lastseen": [],
    }
    for e in entries:
        rows["icao24"].append(e["icao24"])
        rows["callsign"].append(e.get("callsign", ""))
        rows["departure"].append(e.get("departure"))
        rows["arrival"].append(e.get("arrival"))
        rows["typecode"].append(e.get("typecode"))
        rows["firstseen"].append(
            dt(2025, 1, 1, 11, 0, 0, tzinfo=UTC),
        )
        rows["lastseen"].append(
            dt(2025, 1, 1, 13, 0, 0, tzinfo=UTC),
        )

    df = pl.DataFrame(rows)
    download_dir = tmp_path / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    fl_path = download_dir / f"flightlist_{batch_date}.parquet"
    df.write_parquet(fl_path)
    return download_dir


class TestIdentify:
    """Tests for the ``identify`` command (v3 étape 1)."""

    @staticmethod
    def _make_config(tmp_path: Path, *, data_dir: Path) -> Path:
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"
  download_dir: "download"

typecodes:
  - A320
"""
        )
        return config

    def test_identify_flight_id_format(self, tmp_path: Path) -> None:
        """meta_flight_id follows {icao24}_{callsign}_s{idx} format."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Two icao24s, each with a gap → 2 segments each = 4 total segments
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123", "abc123", "def456"],
            callsigns=["TEST01", "TEST01", "FLY02"],
            timestamps_s=[
                list(range(0, 20)),  # abc123 seg 1: 0-19s
                list(range(60, 80)),  # abc123 seg 2: 60-79s (gap > 30s)
                list(range(0, 15)),  # def456 seg 1: 0-14s
            ],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert "meta_flight_id" in result.columns
        assert "meta_original_flight_id" in result.columns

        flight_ids = result["meta_flight_id"].unique().sort().to_list()
        # abc123_TEST01 → 2 segments (_s0, _s1), def456_FLY02 → 1 segment (_s0)
        assert "abc123_TEST01_s0" in flight_ids
        assert "abc123_TEST01_s1" in flight_ids
        assert "def456_FLY02_s0" in flight_ids

        # original_flight_id has no segment suffix
        orig_ids = result["meta_original_flight_id"].unique().sort().to_list()
        assert orig_ids == ["abc123_TEST01", "def456_FLY02"]

    def test_identify_keeps_short_segments(self, tmp_path: Path) -> None:
        """Segments with few points are NOT filtered — they get a flight_id."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 10 points, then gap, then 5 points (short segment)
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123", "abc123"],
            callsigns=["TEST01", "TEST01"],
            timestamps_s=[
                list(range(0, 10)),
                list(range(60, 65)),  # only 5 points — short segment
            ],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        # All 15 rows kept
        assert len(result) == 15
        # Both segments have a flight_id
        ids = result["meta_flight_id"].unique().to_list()
        assert len(ids) == 2

    def test_identify_flightlist_join(self, tmp_path: Path) -> None:
        """Flightlist metadata joined → meta_departure, meta_arrival, meta_aircraft_type."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(
            data_dir,
            entries=[
                {
                    "icao24": "abc123",
                    "callsign": "TEST01",
                    "departure": "LFPG",
                    "arrival": "EGLL",
                    "typecode": "A320",
                },
            ],
        )
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert result["meta_departure"][0] == "LFPG"
        assert result["meta_arrival"][0] == "EGLL"
        assert result["meta_aircraft_type"][0] == "A320"

    def test_identify_no_flightlist(self, tmp_path: Path) -> None:
        """icao24 absent from flightlist → meta_departure/arrival/aircraft_type = null."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(data_dir, entries=[])  # empty flightlist
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        assert result["meta_departure"].null_count() == len(result)
        assert result["meta_arrival"].null_count() == len(result)
        assert result["meta_aircraft_type"].null_count() == len(result)

    def test_identify_gap_below_threshold(self, tmp_path: Path) -> None:
        """Gap of 29s (< 30s threshold) → no split, same segment."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Points at 0-9s, then 38s (29s gap from t=9)
        timestamps = [*range(0, 10), 38]
        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[timestamps],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        ids = result["meta_flight_id"].unique().to_list()
        assert len(ids) == 1  # single segment

    def test_identify_callsign_null(self, tmp_path: Path) -> None:
        """Null callsign → meta_flight_id uses NOCALL placeholder."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=[None],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=False)

        result = pl.read_delta(str(table_path))
        fid = result["meta_flight_id"][0]
        assert "NOCALL" in fid
        assert fid == "abc123_NOCALL_s0"

    def test_identify_dry_run(self, tmp_path: Path) -> None:
        """--dry-run validates config without modifying the Delta Table."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        table_path = _make_delta_table(
            data_dir,
            icao24s=["abc123"],
            callsigns=["TEST01"],
            timestamps_s=[list(range(0, 20))],
        )
        _make_flightlist(data_dir, entries=[])
        config = self._make_config(tmp_path, data_dir=data_dir)

        identify(config=config, dry_run=True)

        result = pl.read_delta(str(table_path))
        assert "meta_flight_id" not in result.columns


# ---------------------------------------------------------------------------
# _split_at_gaps tests (requires traffic)
# ---------------------------------------------------------------------------


def _make_flight(n_points: int, *, gap_at: int | None = None, gap_seconds: int = 60) -> Flight:
    """Create a synthetic Flight with optional data gap.

    Args:
        n_points: Total number of data points.
        gap_at: Index at which to insert a gap.  Points before ``gap_at``
            are 1 s apart; after the gap the first timestamp jumps by
            ``gap_seconds``.
        gap_seconds: Duration of the gap in seconds.
    """
    import numpy as np
    import pandas as pd
    from traffic.core import Flight

    # Build timestamps: 1s apart, with optional gap
    times = []
    t0 = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
    for i in range(n_points):
        if gap_at is not None and i == gap_at:
            t0 += pd.Timedelta(seconds=gap_seconds)
        times.append(t0)
        t0 += pd.Timedelta(seconds=1)

    return Flight(
        pd.DataFrame(
            {
                "timestamp": times,
                "icao24": ["abc123"] * n_points,
                "callsign": ["TEST01"] * n_points,
                "latitude": np.linspace(48.0, 49.0, n_points),
                "longitude": np.linspace(2.0, 3.0, n_points),
                "altitude": np.full(n_points, 35000.0),
            }
        )
    )


class TestSplitAtGaps:
    """Tests for the ``_split_at_gaps`` helper function."""

    @pytest.fixture(autouse=True)
    def _require_traffic(self) -> None:
        pytest.importorskip("traffic")

    def test_split_no_gap(self) -> None:
        """Continuous flight → 1 segment with original_flight_id set."""
        from traffic.core import Traffic

        f = _make_flight(60)
        t = Traffic.from_flights([f])
        assert t is not None
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is not None
        assert len(result) == 1
        seg = next(iter(result))
        assert "original_flight_id" in seg.data.columns
        assert seg.data["original_flight_id"].iloc[0] == "abc123_TEST01"

    def test_split_with_gap(self) -> None:
        """Flight with 60s gap → 2 segments, both with same original_flight_id."""
        from traffic.core import Traffic

        f = _make_flight(80, gap_at=40, gap_seconds=60)
        t = Traffic.from_flights([f])
        assert t is not None
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is not None
        assert len(result) == 2
        segments = list(result)
        # Both segments share the same original flight identity
        ids = {seg.data["original_flight_id"].iloc[0] for seg in segments}
        assert ids == {"abc123_TEST01"}

    def test_split_short_segments_filtered(self) -> None:
        """Segments shorter than min_points are discarded."""
        from traffic.core import Traffic

        # 50 pts total, gap at 5 → seg[0]=5 pts (too short), seg[1]=45 pts (ok)
        f = _make_flight(50, gap_at=5, gap_seconds=60)
        t = Traffic.from_flights([f])
        assert t is not None
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is not None
        assert len(result) == 1  # only the 45-pt segment survives

    def test_split_all_filtered_returns_none(self) -> None:
        """All segments too short → returns None."""
        from traffic.core import Traffic

        # 10 pts, gap at 5 → two segments of 5 pts each, both below min_points=20
        f = _make_flight(10, gap_at=5, gap_seconds=60)
        t = Traffic.from_flights([f])
        assert t is not None
        result = _split_at_gaps(t, threshold="30s", min_points=20)

        assert result is None

    def test_split_empty_traffic(self) -> None:
        """Traffic with only a tiny flight → returns None (all segments too short)."""
        from traffic.core import Traffic

        # 3 points — always below any reasonable min_points
        f = _make_flight(3)
        t = Traffic.from_flights([f])
        assert t is not None
        result = _split_at_gaps(t, threshold="30s", min_points=10)

        assert result is None
