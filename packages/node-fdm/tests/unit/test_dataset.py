"""Unit tests for ``DERIVED_FEATURES`` registration of ``fdm_cl_residual`` (AXM-1737).

Covers AC8: the inverse-PhysicsLayer computer is registered alongside
``fdm_q_pa`` and ``fdm_lift_residual_norm`` and recovers ``cl_residual``
from kinematic columns under the ``m = m_ref`` statistics convention.
"""

from __future__ import annotations

import math

import numpy as np

from node_fdm.dataset import DERIVED_FEATURES
from node_fdm.layers.physics import (
    _M_REF_KG,
    S_REF_A320_M2,
    V_MIN_CLAMP,
    G,
    cl_ref_steady_np,
)


def test_derived_features_includes_cl_residual() -> None:
    """AC8: ``fdm_cl_residual`` is registered in ``DERIVED_FEATURES``."""
    assert "fdm_cl_residual" in DERIVED_FEATURES


def test_compute_cl_residual_inverts_physics_formula() -> None:
    """AC8: inverse computer recovers ``cl_residual`` from ``(gamma, V, d_gamma, alt)``.

    Builds an in-memory batch with known kinematics, evaluates the registered
    computer, and checks the analytic identity
    ``cl_residual = ((V_safe/G · d_gamma + cos gamma) · m_ref · G) / (q · S) - CL_steady(q)``
    on a hand-computed sample. The baseline ``CL_steady(q) = m_ref·g/(q·S)``
    replaces the former constant ``CL_REF=0.5`` so the residual stays
    centered on 0 across phases (post-AXM-1739 fix).
    """
    gamma = 0.05
    tas = 220.0
    d_gamma = 0.002
    alt = 9_000.0
    temp_k = 230.0

    x_cols = ["raw_alt_m", "era_tas_ms", "fdm_gamma_rad"]
    e_cols = ["era_temp_K"]
    dx_cols = ["fdm_d_gamma_rads"]

    n = 3
    x_arr = np.tile(np.array([[alt, tas, gamma]], dtype=np.float64), (n, 1))
    e_arr = np.full((n, 1), temp_k, dtype=np.float64)
    dx_arr = np.full((n, 1), d_gamma, dtype=np.float64)

    fn = DERIVED_FEATURES["fdm_cl_residual"]
    out = fn(x_arr, e_arr, dx_arr, x_cols, e_cols, dx_cols)

    assert out.shape == (n,)
    assert np.isfinite(out).all()

    q_pa = float(DERIVED_FEATURES["fdm_q_pa"](x_arr, e_arr, dx_arr, x_cols, e_cols, dx_cols)[0])
    v_safe = max(tas, V_MIN_CLAMP)
    cl_steady = float(cl_ref_steady_np(np.asarray([q_pa]))[0])
    expected = (((v_safe / G) * d_gamma + math.cos(gamma)) * _M_REF_KG * G) / (
        q_pa * S_REF_A320_M2
    ) - cl_steady
    assert math.isclose(float(out[0]), expected, rel_tol=1e-9)
