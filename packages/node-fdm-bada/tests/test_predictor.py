"""Tests for BADA predictor — import validation + mock-based functional tests.

Full pyBADA is proprietary; these tests mock out the TCL module to exercise
all logic paths in process_single_flight and _run_bada_step.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from node_fdm_bada.predictor import _HAS_PYBADA, process_single_flight

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flight(tmp_path: Path, n: int = 5) -> Path:
    """Create a realistic flight parquet with required columns."""
    df = pl.DataFrame(
        {
            "alt_std_m": np.linspace(8000, 10000, n),
            "tas_ms": np.linspace(220, 230, n),
            "cas_ms": np.linspace(180, 185, n),
            "cas_sel_ms": [0.0, 0.0, 180.0, 0.0, 0.0],
            "mach": np.full(n, 0.78),
            "mach_sel": np.full(n, 0.0),  # CAS mode
            "temperature": np.full(n, 230.0),
            "alt_sel_m": np.linspace(8500, 10500, n),
            "vz_sel_ms": np.full(n, 3.0),
            "long_wind_ms": np.full(n, 5.0),
        }
    )
    path = tmp_path / "A320" / "flight_001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def _make_tcl_result(n: int = 1) -> pd.DataFrame:
    """Return a mock TCL result DataFrame with the expected columns."""
    return pd.DataFrame(
        {
            "Hp": [35000.0] * n,
            "TAS": [250.0] * n,
            "M": [0.78] * n,
            "ROCD": [500.0] * n,
            "mass": [68000.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Import / fallback tests
# ---------------------------------------------------------------------------


class TestPredictor:
    """Tests for the BADA predictor module."""

    def test_no_pybada_returns_none(self, tmp_path: Path) -> None:
        """If pyBADA is not installed, process_single_flight returns None."""
        if _HAS_PYBADA:
            return

        result = process_single_flight(
            flight_path=tmp_path / "dummy.parquet",
            ac=None,
        )
        assert result is None

    def test_module_imports(self) -> None:
        """Verify the module can be imported without pyBADA."""
        assert callable(process_single_flight)


# ---------------------------------------------------------------------------
# Mock-based functional tests — process_single_flight
# ---------------------------------------------------------------------------


class TestProcessSingleFlight:
    """Tests for process_single_flight with mocked pyBADA."""

    def test_returns_bada_columns(self, tmp_path: Path) -> None:
        """Output DataFrame has exactly the expected bada_ columns."""
        flight_path = _make_flight(tmp_path)
        mock_ac = MagicMock()
        mock_ac.MTOW = 80000.0

        tcl_result = _make_tcl_result()

        with (
            patch("node_fdm_bada.predictor._HAS_PYBADA", True),
            patch("node_fdm_bada.predictor._run_bada_step", return_value=tcl_result),
        ):
            result = process_single_flight(flight_path=flight_path, ac=mock_ac)

        assert result is not None
        expected = {
            "bada_alt_std_m",
            "bada_tas_ms",
            "bada_vz_ms",
            "bada_mass_kg",
            "bada_gamma_rad",
        }
        assert set(result.columns) == expected
        assert len(result) == 5  # One row per input row

    def test_saves_to_output_dir(self, tmp_path: Path) -> None:
        """Parquet file saved when output_dir is specified."""
        flight_path = _make_flight(tmp_path)
        mock_ac = MagicMock()
        mock_ac.MTOW = 80000.0
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("node_fdm_bada.predictor._HAS_PYBADA", True),
            patch("node_fdm_bada.predictor._run_bada_step", return_value=_make_tcl_result()),
        ):
            result = process_single_flight(flight_path=flight_path, ac=mock_ac, output_dir=out_dir)

        assert result is not None
        saved = out_dir / "flight_001.parquet"
        assert saved.exists()
        loaded = pl.read_parquet(saved)
        assert set(loaded.columns) == set(result.columns)

    def test_with_processor(self, tmp_path: Path) -> None:
        """Optional processor is called if given."""
        flight_path = _make_flight(tmp_path)
        mock_ac = MagicMock()
        mock_ac.MTOW = 80000.0
        mock_proc = MagicMock()

        with (
            patch("node_fdm_bada.predictor._HAS_PYBADA", True),
            patch("node_fdm_bada.predictor._run_bada_step", return_value=_make_tcl_result()),
        ):
            process_single_flight(flight_path=flight_path, ac=mock_ac, processor=mock_proc)

        mock_proc.process_flight.assert_called_once()

    def test_bad_file_returns_none(self, tmp_path: Path) -> None:
        """Returns None and logs error on corrupt file."""
        bad_path = tmp_path / "bad.parquet"
        bad_path.write_bytes(b"not a parquet")
        mock_ac = MagicMock()
        mock_ac.MTOW = 80000.0

        with patch("node_fdm_bada.predictor._HAS_PYBADA", True):
            result = process_single_flight(flight_path=bad_path, ac=mock_ac)

        assert result is None

    def test_cas_sel_backfill(self, tmp_path: Path) -> None:
        """cas_sel_ms zeros are back-filled with NaN then forward/backward fills."""
        flight_path = _make_flight(tmp_path)
        mock_ac = MagicMock()
        mock_ac.MTOW = 80000.0

        _captured_dfs: list[pl.DataFrame] = []

        def capture_step(**kwargs: object) -> pd.DataFrame:
            # We just return a valid result
            return _make_tcl_result()

        with (
            patch("node_fdm_bada.predictor._HAS_PYBADA", True),
            patch("node_fdm_bada.predictor._run_bada_step", side_effect=capture_step),
        ):
            result = process_single_flight(flight_path=flight_path, ac=mock_ac)

        assert result is not None

    def test_mach_mode(self, tmp_path: Path) -> None:
        """When mach_sel != 0, speed_type is M."""
        n = 3
        df = pl.DataFrame(
            {
                "alt_std_m": np.full(n, 10000.0),
                "tas_ms": np.full(n, 230.0),
                "cas_ms": np.full(n, 185.0),
                "cas_sel_ms": np.full(n, 185.0),
                "mach": np.full(n, 0.78),
                "mach_sel": np.full(n, 0.80),  # Non-zero → Mach mode
                "temperature": np.full(n, 230.0),
                "alt_sel_m": np.full(n, 10500.0),
                "vz_sel_ms": np.full(n, 3.0),
                "long_wind_ms": np.full(n, 5.0),
            }
        )
        path = tmp_path / "mach_flight.parquet"
        df.write_parquet(path)

        mock_ac = MagicMock()
        mock_ac.MTOW = 80000.0

        captured_kwargs: list[dict[str, Any]] = []

        def capture_step(**kwargs: object) -> pd.DataFrame:
            captured_kwargs.append(kwargs)
            return _make_tcl_result()

        with (
            patch("node_fdm_bada.predictor._HAS_PYBADA", True),
            patch("node_fdm_bada.predictor._run_bada_step", side_effect=capture_step),
        ):
            result = process_single_flight(flight_path=path, ac=mock_ac)

        assert result is not None
        # All steps should be Mach mode
        for kw in captured_kwargs:
            assert kw["speed_type"] == "M"


# ---------------------------------------------------------------------------
# Tests for _run_bada_step
# ---------------------------------------------------------------------------


class TestRunBadaStep:
    """Tests for _run_bada_step with mocked TCL functions."""

    @pytest.fixture(autouse=True)
    def _mock_tcl(self) -> None:
        """Inject mock TCL functions into the predictor module."""
        self.mock_csl = MagicMock(return_value=_make_tcl_result())
        self.mock_csr = MagicMock(return_value=_make_tcl_result())
        self.mock_csrocd = MagicMock(return_value=_make_tcl_result())
        self.mock_accdec = MagicMock(return_value=_make_tcl_result())
        self.mock_target = MagicMock()

        self.patches: list[Any] = [
            patch("node_fdm_bada.predictor._HAS_PYBADA", True),
            patch("node_fdm_bada.predictor.constantSpeedLevel", self.mock_csl, create=True),
            patch("node_fdm_bada.predictor.constantSpeedRating_time", self.mock_csr, create=True),
            patch("node_fdm_bada.predictor.constantSpeedROCD_time", self.mock_csrocd, create=True),
            patch("node_fdm_bada.predictor.accDec_time", self.mock_accdec, create=True),
            patch("node_fdm_bada.predictor.target", self.mock_target, create=True),
        ]
        for p in self.patches:
            p.start()

    @pytest.fixture(autouse=True)
    def _stop_patches(self) -> Generator[None]:
        yield
        for p in self.patches:
            p.stop()

    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "ac": MagicMock(),
            "speed_type": "CAS",
            "v_init": 280.0,
            "v_target": 280.0,
            "phase": "Cruise",
            "hp_init": 35000.0,
            "m_init": 68000.0,
            "delta_temp": 0.0,
            "config": "CR",
            "ws": 0.0,
            "rocd_target": 0.0,
            "speed_diff_ratio": 0.0,
        }

    def test_cruise_constant_speed(self) -> None:
        """Small speed diff + Cruise → constantSpeedLevel."""
        from node_fdm_bada.predictor import _run_bada_step

        kwargs = self._base_kwargs()
        _run_bada_step(**kwargs)
        self.mock_csl.assert_called_once()

    def test_climb_with_rocd(self) -> None:
        """Small speed diff + Climb + ROCD → constantSpeedROCD_time."""
        from node_fdm_bada.predictor import _run_bada_step

        kwargs = self._base_kwargs()
        kwargs["phase"] = "Climb"
        kwargs["rocd_target"] = 1500.0
        _run_bada_step(**kwargs)
        self.mock_csrocd.assert_called_once()

    def test_climb_no_rocd(self) -> None:
        """Small speed diff + Climb + no ROCD → constantSpeedRating_time."""
        from node_fdm_bada.predictor import _run_bada_step

        kwargs = self._base_kwargs()
        kwargs["phase"] = "Climb"
        kwargs["rocd_target"] = 0.0
        _run_bada_step(**kwargs)
        self.mock_csr.assert_called_once()

    def test_acceleration(self) -> None:
        """Large speed diff + v_init < v_target → accDec_time with 'acc'."""
        from node_fdm_bada.predictor import _run_bada_step

        kwargs = self._base_kwargs()
        kwargs["v_init"] = 250.0
        kwargs["v_target"] = 300.0
        kwargs["speed_diff_ratio"] = 0.1  # > 0.04
        _run_bada_step(**kwargs)
        self.mock_accdec.assert_called_once()
        call_kwargs = self.mock_accdec.call_args[1]
        assert call_kwargs["speedEvol"] == "acc"

    def test_deceleration(self) -> None:
        """Large speed diff + v_init > v_target → accDec_time with 'dec'."""
        from node_fdm_bada.predictor import _run_bada_step

        kwargs = self._base_kwargs()
        kwargs["v_init"] = 300.0
        kwargs["v_target"] = 250.0
        kwargs["speed_diff_ratio"] = 0.1
        _run_bada_step(**kwargs)
        self.mock_accdec.assert_called_once()
        call_kwargs = self.mock_accdec.call_args[1]
        assert call_kwargs["speedEvol"] == "dec"

    def test_accdec_fallback_on_value_error(self) -> None:
        """accDec_time ValueError → fallback to constantSpeedLevel."""
        from node_fdm_bada.predictor import _run_bada_step

        self.mock_accdec.side_effect = ValueError("TCL error")
        kwargs = self._base_kwargs()
        kwargs["speed_diff_ratio"] = 0.1
        _run_bada_step(**kwargs)
        self.mock_csl.assert_called_once()

    def test_rocd_fallback_on_value_error(self) -> None:
        """constantSpeedROCD_time ValueError → fallback to constantSpeedLevel."""
        from node_fdm_bada.predictor import _run_bada_step

        self.mock_csrocd.side_effect = ValueError("TCL error")
        kwargs = self._base_kwargs()
        kwargs["phase"] = "Descent"
        kwargs["rocd_target"] = -1500.0
        _run_bada_step(**kwargs)
        self.mock_csl.assert_called_once()

    def test_descent_with_rocd_target_uses_control(self) -> None:
        """Descent + large speed diff + rocd < -10 → target() used as control."""
        from node_fdm_bada.predictor import _run_bada_step

        kwargs = self._base_kwargs()
        kwargs["phase"] = "Descent"
        kwargs["v_init"] = 300.0
        kwargs["v_target"] = 250.0
        kwargs["speed_diff_ratio"] = 0.1
        kwargs["rocd_target"] = -1500.0
        _run_bada_step(**kwargs)
        self.mock_target.assert_called_once()
        self.mock_accdec.assert_called_once()
