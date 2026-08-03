from __future__ import annotations

from _config_fixtures import SELECTED_PARAMS
from node_fdm_pipeline.config import GammaFilterConfig, VzFilterConfig


def test_gamma_filter_config_default_mode_bilateral() -> None:
    """``mode`` still defaults to bilateral; the tuning must now be declared.

    Previously asserted defaults for sigma_s/sigma_r/slope_tol/flat_tol/
    abs_min/min_len. Those fields no longer carry defaults, so the assertions
    check the declared calibration round-trips instead.
    """
    cfg = GammaFilterConfig(**SELECTED_PARAMS["gamma"])  # type: ignore[arg-type]
    assert cfg.mode == "bilateral_gamma"
    assert cfg.sigma_s == 6.0
    assert cfg.sigma_r == 0.002
    assert cfg.slope_tol == 1.2e-3
    assert cfg.flat_tol == 2e-3
    assert cfg.abs_min == 5e-3
    assert cfg.min_len == 10


def test_vz_filter_config_default_mode_bilateral() -> None:
    """``mode`` still defaults to bilateral; the tuning must now be declared.

    The old assertion here was ``sigma_r == 350.0`` — the superseded value that
    motivated making these fields required (opensky26 tbl.3 retained 100.0).
    """
    cfg = VzFilterConfig(**SELECTED_PARAMS["vz"])  # type: ignore[arg-type]
    assert cfg.mode == "bilateral_vz"
    assert cfg.sigma_s == 6.0
    assert cfg.sigma_r == 100.0
    assert cfg.slope_tol == 50.0
    assert cfg.flat_tol == 100.0
    assert cfg.min_len == 10
