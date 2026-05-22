"""Phase 4 — AC2 grid parity: PSEfficiencyLayer vs ps_engine reference.

Validates that the torch port of P&S Part 3 Eqs 24-29 + Appendix C1-C5
reproduces the numpy ps_engine reference within 2% across a 25-point
high-thrust grid and 6 explicit low-thrust points.

Pattern mirror of scripts/phase3_thrust_ac2_grid.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Lazy import of ps_engine (lib lives outside the repo).
_PS_LIB_ROOT = "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages"
for _pkg in ("ps-engine", "ps-core", "ps-aircraft", "ps-flight"):
    _src = f"{_PS_LIB_ROOT}/{_pkg}/src"
    if _src not in sys.path:
        sys.path.insert(0, _src)

from ps_engine.database import get_engine_params  # type: ignore[import-not-found]
from ps_engine.efficiency import eta_o, eta_o_with_idle_fallback  # type: ignore[import-not-found]

# Resolve node-fdm src on path.
REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_FDM_SRC = REPO_ROOT / "packages" / "node-fdm" / "src"
if str(NODE_FDM_SRC) not in sys.path:
    sys.path.insert(0, str(NODE_FDM_SRC))

from node_fdm.layers.ps_efficiency import (  # noqa: E402
    PSEfficiencyLayer,
    compute_a320_efficiency_constants,
    eta_o_reference,
)

OUT_PATH = REPO_ROOT / "data" / "investigations" / "phase4_fuelflow" / "artifacts" / "ac2_eta_grid.md"


def _ratio_to_cutoff(c_t: float, mach: float, k_c_t_ratio: float, m_do: float, c_t_do: float) -> float:
    """Compute C_T / (C_T)_etaB for the given (c_t, mach) — used to decide cubic vs Eq24."""
    h_2 = ((1.0 + k_c_t_ratio * mach) / (1.0 + k_c_t_ratio * m_do)) * (m_do / mach) ** 2
    c_t_eta_b = h_2 * c_t_do
    return c_t / c_t_eta_b


def main() -> None:  # noqa: PLR0915
    consts = compute_a320_efficiency_constants()
    engine = get_engine_params("A320")
    layer_cruise = PSEfficiencyLayer(mode="cruise").eval()
    layer_fallback = PSEfficiencyLayer(mode="with_idle_fallback").eval()

    print(f"[constants] {consts}")
    print(f"[engine_params] {engine.model_dump()}")

    high_thrust_c_t = [0.005, 0.015, 0.025, 0.035, 0.045]
    high_thrust_m = [0.5, 0.6, 0.7, 0.78, 0.82]
    low_thrust_c_t = [0.003, 0.007, 0.010]
    low_thrust_m = [0.6, 0.78]

    rows: list[dict[str, object]] = []
    abs_errs_high: list[float] = []
    abs_errs_low: list[float] = []

    # === High-thrust grid (Eq 24) ===
    for c_t in high_thrust_c_t:
        for mach in high_thrust_m:
            ratio = _ratio_to_cutoff(c_t, mach, consts["C_T_ratio_K"], consts["M_DO"], consts["C_T_DO"])
            branch = "low" if ratio < consts["low_thrust_cutoff"] else "high"

            # Reference (numpy) -- Eq 24 directly (no fallback at any cost on this grid).
            eta_ref = float(eta_o(c_t, mach, engine))

            # Torch (cruise mode -- always Eq 24).
            c_t_t = torch.tensor([c_t], dtype=torch.float32)
            m_t = torch.tensor([mach], dtype=torch.float32)
            eta_torch = float(layer_cruise(c_t_t, m_t).item())

            # Scalar reference (no fallback, mirrors cruise mode).
            eta_scalar = float(eta_o_reference(c_t, mach, use_low_thrust_fallback=False))

            err = (eta_torch - eta_ref) / eta_ref * 100.0
            err_scalar = (eta_scalar - eta_ref) / eta_ref * 100.0
            rows.append(
                {
                    "C_T": c_t,
                    "M": mach,
                    "ratio": ratio,
                    "branch_expected": branch,
                    "eta_ref": eta_ref,
                    "eta_torch": eta_torch,
                    "eta_scalar": eta_scalar,
                    "err_pct": err,
                    "err_scalar_pct": err_scalar,
                    "grid": "high",
                }
            )
            abs_errs_high.append(abs(err))

    # === Low-thrust grid (Appendix C) ===
    for c_t in low_thrust_c_t:
        for mach in low_thrust_m:
            ratio = _ratio_to_cutoff(c_t, mach, consts["C_T_ratio_K"], consts["M_DO"], consts["C_T_DO"])
            branch = "low" if ratio < consts["low_thrust_cutoff"] else "high"

            # Reference uses fallback API.
            eta_ref = float(eta_o_with_idle_fallback(c_t, mach, engine))

            # Torch fallback layer.
            c_t_t = torch.tensor([c_t], dtype=torch.float32)
            m_t = torch.tensor([mach], dtype=torch.float32)
            eta_torch = float(layer_fallback(c_t_t, m_t).item())

            eta_scalar = float(eta_o_reference(c_t, mach, use_low_thrust_fallback=True))

            err = (eta_torch - eta_ref) / eta_ref * 100.0
            err_scalar = (eta_scalar - eta_ref) / eta_ref * 100.0
            rows.append(
                {
                    "C_T": c_t,
                    "M": mach,
                    "ratio": ratio,
                    "branch_expected": branch,
                    "eta_ref": eta_ref,
                    "eta_torch": eta_torch,
                    "eta_scalar": eta_scalar,
                    "err_pct": err,
                    "err_scalar_pct": err_scalar,
                    "grid": "low",
                }
            )
            abs_errs_low.append(abs(err))

    max_err_high = max(abs_errs_high) if abs_errs_high else 0.0
    max_err_low = max(abs_errs_low) if abs_errs_low else 0.0
    max_err_total = max(max_err_high, max_err_low)
    verdict = "PASS" if max_err_total < 2.0 else "FAIL"

    # === Output markdown ===
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write("# AC2 — PSEfficiencyLayer parity vs ps_engine.efficiency reference\n\n")
        f.write(f"Generated by `scripts/phase4_fuelflow_eta_grid.py`.\n\n")
        f.write(f"**Verdict** : {verdict}  (target < 2 %)\n\n")
        f.write(f"- High-thrust grid (Eq 24) max abs err : **{max_err_high:.4f} %** (n={len(abs_errs_high)})\n")
        f.write(f"- Low-thrust grid (Appendix C) max abs err : **{max_err_low:.4f} %** (n={len(abs_errs_low)})\n")
        f.write(f"- Total max abs err : **{max_err_total:.4f} %**\n\n")
        f.write("## Constants used (from ps_engine database)\n\n")
        for k, v in consts.items():
            f.write(f"- `{k}` = {v}\n")
        f.write("\n## Grid samples\n\n")
        f.write("| Grid | C_T | M | ratio=C_T/(C_T)_etaB | branch | eta_ref | eta_torch | eta_scalar | err_torch (%) | err_scalar (%) |\n")
        f.write("|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['grid']} | {r['C_T']:.4f} | {r['M']:.3f} | {r['ratio']:.4f} | {r['branch_expected']} | "
                f"{r['eta_ref']:.6f} | {r['eta_torch']:.6f} | {r['eta_scalar']:.6f} | "
                f"{r['err_pct']:+.4f} | {r['err_scalar_pct']:+.4f} |\n"
            )
        f.write("\n## Notes\n\n")
        f.write("- High-thrust grid evaluates `eta_o(C_T, M)` via Eq 24 (no fallback).\n")
        f.write("- Low-thrust grid evaluates `eta_o_with_idle_fallback(C_T, M)`. Branch column shows what the cubic vs Eq24 split says for that point.\n")
        f.write("- `err_torch` = (eta_torch - eta_ref) / eta_ref * 100. Compared via float32 forward.\n")
        f.write("- `err_scalar` = same but for the numpy `eta_o_reference` scalar helper (validates the helper too).\n")

    print(f"\n[verdict] max abs err total = {max_err_total:.4f} %  ({verdict})")
    print(f"           max abs err high  = {max_err_high:.4f} %  (Eq 24)")
    print(f"           max abs err low   = {max_err_low:.4f} %  (Appendix C)")
    print(f"\nReport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
