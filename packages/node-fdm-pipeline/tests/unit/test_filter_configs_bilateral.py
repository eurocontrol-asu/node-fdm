from __future__ import annotations

from node_fdm_pipeline.config import GammaFilterConfig, VzFilterConfig


def test_gamma_filter_config_default_mode_bilateral() -> None:
    cfg = GammaFilterConfig()
    assert cfg.mode == "bilateral_gamma"
    assert cfg.sigma_s == 6.0
    assert cfg.sigma_r == 1.2e-2
    assert cfg.slope_tol == 3e-4
    assert cfg.flat_tol == 2e-3
    assert cfg.abs_min == 5e-3
    assert cfg.min_len == 10


def test_vz_filter_config_default_mode_bilateral() -> None:
    cfg = VzFilterConfig()
    assert cfg.mode == "bilateral_vz"
    assert cfg.sigma_s == 6.0
    assert cfg.sigma_r == 350.0
    assert cfg.slope_tol == 15.0
    assert cfg.flat_tol == 100.0
    assert cfg.min_len == 10
