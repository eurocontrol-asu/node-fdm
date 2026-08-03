"""Shared test fixtures for node-fdm-pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


#: The detector tuning every test config needs, since ``SelectedParamConfig``
#: makes its five channels required (see the note in ``config.py``).
#:
#: These are the values paper_opensky26 retained on its coverage x
#: self-consistency Pareto front over 1,472 flights — main.tex table 3 — not
#: arbitrary filler. That matters: a fixture holding invented numbers would be
#: the very default the required fields exist to abolish, only hidden one level
#: deeper. Tuning that breaks a test here breaks it against a published
#: calibration, which is a statement worth reading.
#:
#: The channels quote sigma_r and slope-tol from the paper. The remaining knobs
#: (sigma_s, n_passes, flat_tol, min_len, cutoff_s) are the pipeline's own,
#: outside the swept grid, and are carried over from the values these models
#: shipped with.
SELECTED_PARAMS_YAML = """\
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""

#: The same block as nested dicts, for tests building a config in Python.
SELECTED_PARAMS: dict[str, dict[str, float | int]] = {
    "mach": {
        "sigma_s": 8.0,
        "sigma_r": 0.01,
        "n_passes": 2,
        "slope_tol": 3.0e-4,
        "flat_tol": 5.0e-2,
        "min_len": 15,
    },
    "cas": {
        "cutoff_s": 180.0,
        "sigma_s": 8.0,
        "sigma_r": 3.0,
        "n_passes": 2,
        "slope_tol": 0.09,
        "flat_tol": 20.0,
        "min_len": 5,
    },
    "vz": {
        "sigma_s": 6.0,
        "sigma_r": 100.0,
        "slope_tol": 50.0,
        "flat_tol": 100.0,
        "min_len": 10,
    },
    "alt": {
        "sigma_s": 6.0,
        "sigma_r": 20.0,
        "n_passes": 2,
        "tol_ftmin": 150.0,
        "min_len": 6,
    },
    "gamma": {
        "sigma_s": 6.0,
        "sigma_r": 0.002,
        "slope_tol": 1.2e-3,
        "flat_tol": 2.0e-3,
        "abs_min": 5.0e-3,
        "min_len": 10,
    },
}


@pytest.fixture
def mock_fastmeteo(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Isolated fastmeteo sys.modules mock with strict per-test cleanup.

    Uses ``monkeypatch.setitem`` instead of ``mocker.patch.dict`` so that
    each test gets fresh mocks and ``sys.modules`` is fully restored on
    teardown — preventing cross-test import cache contamination.

    Returns ``(mock_arco_cls, mock_arco_instance)`` so tests can configure
    ``mock_arco_instance.interpolate.side_effect`` as needed.
    """
    mock_arco_cls = MagicMock()
    mock_arco_instance = MagicMock()
    mock_arco_cls.return_value = mock_arco_instance

    mock_source_arco = MagicMock(ArcoEra5=mock_arco_cls)
    mock_source = MagicMock(arco_era5=mock_source_arco)

    monkeypatch.setitem(sys.modules, "fastmeteo", MagicMock())
    monkeypatch.setitem(sys.modules, "fastmeteo.source", mock_source)
    monkeypatch.setitem(sys.modules, "fastmeteo.source.arco_era5", mock_source_arco)

    return mock_arco_cls, mock_arco_instance


@pytest.fixture
def opensky_config_path(tmp_path: Path) -> Path:
    """Create a realistic OpenSky config YAML (replaces deleted scripts/opensky/config.yaml)."""
    config = tmp_path / "config.yaml"
    config.write_text("""\
paths:
  data_dir: "TODO"
  preprocess_dir: "preprocessed_parquet"
  era5_cache_dir: "era5_cache"

era5_features:
  - u_component_of_wind
  - v_component_of_wind
  - temperature

typecodes:
  - A320
  - A20N
  - A321
  - A21N
  - B738
  - B38M
  - E190
  - E195
  - A319
  - A332
  - A333
selected_params:
  mach: {sigma_s: 8.0, sigma_r: 0.01, n_passes: 2, slope_tol: 3.0e-4, flat_tol: 5.0e-2, min_len: 15}
  cas: {cutoff_s: 180.0, sigma_s: 8.0, sigma_r: 3.0, n_passes: 2, slope_tol: 0.09, flat_tol: 20.0, min_len: 5}
  vz: {sigma_s: 6.0, sigma_r: 100.0, slope_tol: 50.0, flat_tol: 100.0, min_len: 10}
  alt: {sigma_s: 6.0, sigma_r: 20.0, n_passes: 2, tol_ftmin: 150.0, min_len: 6}
  gamma: {sigma_s: 6.0, sigma_r: 0.002, slope_tol: 1.2e-3, flat_tol: 2.0e-3, abs_min: 5.0e-3, min_len: 10}

computing:
  default_cpu_count: 35

bada:
  bada_4_2_dir: "/path/to/BADA/4.2.1"
""")
    return config


@pytest.fixture
def qar_config_path(tmp_path: Path) -> Path:
    """Create a minimal QAR config YAML (replaces deleted scripts/qar/config.yaml)."""
    config = tmp_path / "config.yaml"
    config.write_text("""\
paths:
  data_dir: "TODO"

typecodes:
  - A320
selected_params:
  mach: {sigma_s: 8.0, sigma_r: 0.01, n_passes: 2, slope_tol: 3.0e-4, flat_tol: 5.0e-2, min_len: 15}
  cas: {cutoff_s: 180.0, sigma_s: 8.0, sigma_r: 3.0, n_passes: 2, slope_tol: 0.09, flat_tol: 20.0, min_len: 5}
  vz: {sigma_s: 6.0, sigma_r: 100.0, slope_tol: 50.0, flat_tol: 100.0, min_len: 10}
  alt: {sigma_s: 6.0, sigma_r: 20.0, n_passes: 2, tol_ftmin: 150.0, min_len: 6}
  gamma: {sigma_s: 6.0, sigma_r: 0.002, slope_tol: 1.2e-3, flat_tol: 2.0e-3, abs_min: 5.0e-3, min_len: 10}
""")
    return config


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal valid config YAML in a temp directory."""
    data_dir = tmp_path / "test_data"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
selected_params:
  mach: {sigma_s: 8.0, sigma_r: 0.01, n_passes: 2, slope_tol: 3.0e-4, flat_tol: 5.0e-2, min_len: 15}
  cas: {cutoff_s: 180.0, sigma_s: 8.0, sigma_r: 3.0, n_passes: 2, slope_tol: 0.09, flat_tol: 20.0, min_len: 5}
  vz: {sigma_s: 6.0, sigma_r: 100.0, slope_tol: 50.0, flat_tol: 100.0, min_len: 10}
  alt: {sigma_s: 6.0, sigma_r: 20.0, n_passes: 2, tol_ftmin: 150.0, min_len: 6}
  gamma: {sigma_s: 6.0, sigma_r: 0.002, slope_tol: 1.2e-3, flat_tol: 2.0e-3, abs_min: 5.0e-3, min_len: 10}
"""
    )
    return config


@pytest.fixture
def empty_typecodes_config(tmp_path: Path) -> Path:
    """Create a config YAML with empty typecodes (should fail validation)."""
    data_dir = tmp_path / "test_data"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""\
paths:
  data_dir: "{data_dir}"

typecodes: []
selected_params:
  mach: {sigma_s: 8.0, sigma_r: 0.01, n_passes: 2, slope_tol: 3.0e-4, flat_tol: 5.0e-2, min_len: 15}
  cas: {cutoff_s: 180.0, sigma_s: 8.0, sigma_r: 3.0, n_passes: 2, slope_tol: 0.09, flat_tol: 20.0, min_len: 5}
  vz: {sigma_s: 6.0, sigma_r: 100.0, slope_tol: 50.0, flat_tol: 100.0, min_len: 10}
  alt: {sigma_s: 6.0, sigma_r: 20.0, n_passes: 2, tol_ftmin: 150.0, min_len: 6}
  gamma: {sigma_s: 6.0, sigma_r: 0.002, slope_tol: 1.2e-3, flat_tol: 2.0e-3, abs_min: 5.0e-3, min_len: 10}
"""
    )
    return config
