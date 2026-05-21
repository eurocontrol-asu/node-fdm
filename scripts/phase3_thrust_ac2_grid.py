"""AC2 grid parity for PSThrustLayer (Exp 11 of Phase 3).

Verifies that the torch ``PSThrustLayer`` reproduces the same forward
thrust composition as a numpy reference built from ``ps_engine``
primitives (``_h2`` of Eq 28 + database constants).

The two implementations share the same equations and constants, so the
parity check is essentially a sanity test on torch dtypes / clamp
semantics. We accept it because (a) Eq 28 is exact P&S, (b) the
constants come from the same ``ps_engine.database`` Table 1 + Table 2,
and (c) the AC2 budget on this Phase 3 layer is < 5 % per the ticket.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure ps_engine + ps_core + ps_aircraft are importable.
_PS_LIB_ROOT = Path(
    "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages"
)
for _pkg in ("ps-core", "ps-aircraft", "ps-engine", "ps-flight"):
    _src = str(_PS_LIB_ROOT / _pkg / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from ps_engine.database import get_engine_params  # type: ignore[import-not-found]  # noqa: E402
from ps_engine.efficiency import _h2 as ps_h2  # type: ignore[import-not-found]  # noqa: E402

from node_fdm.layers.physics import S_REF_A320_M2  # noqa: E402
from node_fdm.layers.ps_thrust import (  # noqa: E402
    PSThrustLayer,
    compute_a320_thrust_constants,
    t_ps_reference,
)

# ISA atmosphere helpers (sufficient for cruise FL/M sweep).
_T_SL_K = 288.15
_P_SL_PA = 101325.0
_RHO_SL = 1.225
_L_K_PER_M = 0.0065
_TROP_M = 11_000.0
_T_TROP_K = _T_SL_K - _L_K_PER_M * _TROP_M
_P_TROP_PA = _P_SL_PA * (_T_TROP_K / _T_SL_K) ** 5.2561
_GAMMA = 1.4
_R_GAS = 287.05


def isa_state(alt_m: float, mach: float) -> tuple[float, float, float, float]:
    """Return (T_K, p_Pa, rho, V_ms) for ISA at altitude + Mach."""
    if alt_m < _TROP_M:
        t_k = _T_SL_K - _L_K_PER_M * alt_m
        p_pa = _P_SL_PA * (t_k / _T_SL_K) ** 5.2561
    else:
        t_k = _T_TROP_K
        p_pa = _P_TROP_PA * math.exp(-9.80665 * (alt_m - _TROP_M) / (_R_GAS * _T_TROP_K))
    rho = p_pa / (_R_GAS * t_k)
    a = math.sqrt(_GAMMA * _R_GAS * t_k)
    v = mach * a
    return t_k, p_pa, rho, v


def t_ps_ref_via_ps_engine(throttle: float, mach: float, q_pa: float) -> float:
    """Reference T_total via ps_engine._h2 (Eq 28) + database constants.

    This is the *external* reference — uses ps_engine's own _h2 implementation
    rather than the local copy in ps_thrust.t_ps_reference, so it independently
    verifies that the torch layer aligns with the upstream library.
    """
    engine = get_engine_params("A320")
    c_t_eta_b = ps_h2(mach, engine.M_DO) * engine.C_T_DO
    chi = min(max(throttle, 0.0), 1.0)
    c_t_inst = chi * c_t_eta_b
    return c_t_inst * q_pa * S_REF_A320_M2


def main() -> None:
    layer = PSThrustLayer()
    layer.eval()

    constants = compute_a320_thrust_constants()
    print("PSThrustLayer constants (from ps_engine database):")
    for k, v in constants.items():
        print(f"  {k:14s} = {v}")
    print()

    FLs = [310, 350, 370, 390, 410]
    Machs = [0.70, 0.74, 0.78, 0.82]
    throttles = [0.4, 0.6, 0.8, 1.0]

    rows: list[dict[str, float]] = []
    for fl in FLs:
        alt_m = fl * 100.0 * 0.3048
        for m in Machs:
            t_k, _p_pa, rho, v = isa_state(alt_m, m)
            q_pa = 0.5 * rho * v * v
            for chi in throttles:
                # Torch layer
                torch_t = layer(
                    torch.tensor([chi], dtype=torch.float32),
                    torch.tensor([m], dtype=torch.float32),
                    torch.tensor([alt_m], dtype=torch.float32),
                    torch.tensor([t_k], dtype=torch.float32),
                    torch.tensor([q_pa], dtype=torch.float32),
                ).item()
                # External reference via ps_engine
                ref_ext = t_ps_ref_via_ps_engine(chi, m, q_pa)
                # Internal reference (numpy mirror of the layer's equations)
                ref_int = t_ps_reference(chi, m, q_pa)
                err_ext = (torch_t - ref_ext) / max(ref_ext, 1.0) * 100.0
                err_int = (torch_t - ref_int) / max(ref_int, 1.0) * 100.0
                rows.append(
                    {
                        "FL": float(fl),
                        "M": m,
                        "chi": chi,
                        "q_pa": q_pa,
                        "T_torch_kN": torch_t / 1000.0,
                        "T_ref_kN": ref_ext / 1000.0,
                        "err_vs_ext_%": err_ext,
                        "err_vs_int_%": err_int,
                    }
                )

    abs_errs_ext = np.array([abs(r["err_vs_ext_%"]) for r in rows])
    abs_errs_int = np.array([abs(r["err_vs_int_%"]) for r in rows])
    print(f"Grid points : {len(rows)}")
    print("Parity vs external ps_engine reference (Eq 28 via ps_engine._h2):")
    print(f"  mean |err| = {abs_errs_ext.mean():.4f} %")
    print(f"  max  |err| = {abs_errs_ext.max():.4f} %")
    print("Parity vs internal numpy mirror (sanity check):")
    print(f"  mean |err| = {abs_errs_int.mean():.4f} %")
    print(f"  max  |err| = {abs_errs_int.max():.4f} %")
    print()
    # Show extremes
    worst = max(rows, key=lambda r: abs(r["err_vs_ext_%"]))
    print(
        f"Worst point : FL{int(worst['FL'])} M={worst['M']:.2f} chi={worst['chi']:.1f}"
        f" -> err={worst['err_vs_ext_%']:+.3f} % (T_torch={worst['T_torch_kN']:.2f} kN,"
        f" T_ref={worst['T_ref_kN']:.2f} kN)"
    )

    # Headline table — cruise corner samples.
    print()
    print("Sample grid (cruise corners):")
    print(f"{'FL':>4} {'M':>5} {'chi':>5} {'q(kPa)':>8} {'T_torch':>10} {'T_ref':>10} {'err%':>8}")
    for r in rows[::6]:
        print(
            f"{int(r['FL']):>4} {r['M']:>5.2f} {r['chi']:>5.2f} "
            f"{r['q_pa']/1000:>8.2f} {r['T_torch_kN']:>10.3f} {r['T_ref_kN']:>10.3f} "
            f"{r['err_vs_ext_%']:>+8.4f}"
        )

    # Verdict
    AC2_BUDGET_PCT = 5.0
    if abs_errs_ext.max() < AC2_BUDGET_PCT:
        print()
        print(f"AC2 PASS : max err {abs_errs_ext.max():.3f}% < {AC2_BUDGET_PCT}% budget.")
    else:
        print()
        print(f"AC2 FAIL : max err {abs_errs_ext.max():.3f}% >= {AC2_BUDGET_PCT}% budget.")


if __name__ == "__main__":
    main()
