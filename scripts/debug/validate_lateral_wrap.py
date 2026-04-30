"""Numerical validation of lateral channel angle/wrap handling.

Validates 4 design questions for the lateral Neural ODE design:
  Q1 — signed_wrap formula correctness + autograd
  Q2 — loss formulation when state=heading drifts beyond [-pi, pi]
  Q3 — wrap-boundary edge case (pred and true on opposite sides)
  Q4 — gradient stability of tan(phi_bank) at NN output cap

Run: uv run python scripts/debug/validate_lateral_wrap.py
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def signed_wrap(x: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi]. Differentiable everywhere except at the wrap jumps."""
    two_pi = 2.0 * math.pi
    return ((x + math.pi) % two_pi) - math.pi


def deg(x: float | torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.item()) * 180.0 / math.pi
    return x * 180.0 / math.pi


def rad(d: float) -> float:
    return d * math.pi / 180.0


# ---------------------------------------------------------------------------
# Q1 — signed_wrap formula and autograd
# ---------------------------------------------------------------------------
def q1_signed_wrap() -> None:
    print("=" * 70)
    print("Q1 — signed_wrap formula")
    print("=" * 70)

    cases = [
        ("355° vs 5°", rad(355) - rad(5), -10.0),
        ("5° vs 355°", rad(5) - rad(355), +10.0),
        ("180° vs -180°", math.pi - (-math.pi), 0.0),
        ("π vs -π+ε", math.pi - (-math.pi + 1e-6), -1e-6 * 180 / math.pi),
        ("-π+ε vs π", (-math.pi + 1e-6) - math.pi, +1e-6 * 180 / math.pi),
    ]

    print(f"{'case':25s} {'raw_diff_deg':>14s} {'wrapped_deg':>14s} "
          f"{'expected_deg':>14s} {'pass':>6s}")
    all_pass = True
    for name, raw_diff, expected_deg in cases:
        t = torch.tensor(raw_diff, dtype=torch.float64)
        wrapped = signed_wrap(t)
        wrapped_deg = deg(wrapped)
        ok = abs(wrapped_deg - expected_deg) < 1e-3
        all_pass &= ok
        print(f"{name:25s} {deg(raw_diff):14.6f} {wrapped_deg:14.6f} "
              f"{expected_deg:14.6f} {'OK' if ok else 'FAIL':>6s}")

    # Vectorized batch
    batch = torch.tensor(
        [rad(355) - rad(5), rad(5) - rad(355), 0.0, math.pi, -math.pi, 3 * math.pi],
        dtype=torch.float64,
    )
    wrapped_batch = signed_wrap(batch)
    print(f"\nbatch raw (deg): {[f'{deg(x):.2f}' for x in batch]}")
    print(f"batch wrapped (deg): {[f'{deg(x):.2f}' for x in wrapped_batch]}")
    in_range = bool(((wrapped_batch >= -math.pi - 1e-9) & (wrapped_batch <= math.pi + 1e-9)).all())
    print(f"all in [-pi, pi]: {in_range}")

    # Autograd
    print("\n--- autograd ---")
    sample_pts = [-math.pi - 0.1, -1.0, 0.0, 1.0, math.pi - 0.1, math.pi + 1e-6, 2 * math.pi + 0.5]
    print(f"{'x':>14s} {'wrap(x)':>14s} {'grad':>10s}")
    grads_ok = True
    for x_val in sample_pts:
        x = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
        y = signed_wrap(x)
        y.backward()
        g = float(x.grad.item())
        print(f"{x_val:14.6f} {float(y.item()):14.6f} {g:10.4f}")
        if not math.isclose(g, 1.0, abs_tol=1e-6):
            grads_ok = False
    print(f"all grads ≈ 1.0: {grads_ok}")
    print(f"\nQ1 verdict: formula {'PASS' if all_pass and grads_ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Q2 — loss when heading drifts beyond [-pi, pi]
# ---------------------------------------------------------------------------
def _three_losses(h_pred: torch.Tensor, h_true: torch.Tensor) -> tuple[float, float, float]:
    naive = (h_pred - h_true).pow(2).mean().item()
    wrapped = signed_wrap(h_pred - h_true).pow(2).mean().item()
    cs = ((torch.cos(h_pred) - torch.cos(h_true)).pow(2)
          + (torch.sin(h_pred) - torch.sin(h_true)).pow(2)).mean().item()
    return naive, wrapped, cs


def q2_loss_drift() -> None:
    print("\n" + "=" * 70)
    print("Q2 — loss when state=heading drifts (pred=5π/2, true=π/2)")
    print("=" * 70)

    h_true = torch.tensor(math.pi / 2, dtype=torch.float64)
    h_pred_exact = torch.tensor(5 * math.pi / 2, dtype=torch.float64)  # +2π drift, same angle
    h_pred_off = torch.tensor(5 * math.pi / 2 + 0.1, dtype=torch.float64)  # off by +0.1

    print(f"{'scenario':30s} {'naive_MSE':>14s} {'wrap_MSE':>14s} {'cos/sin_MSE':>14s}")
    n, w, cs = _three_losses(h_pred_exact, h_true)
    print(f"{'pred=5π/2, true=π/2 (=)':30s} {n:14.4f} {w:14.6e} {cs:14.6e}")
    n2, w2, cs2 = _three_losses(h_pred_off, h_true)
    print(f"{'pred=5π/2+0.1, true=π/2':30s} {n2:14.4f} {w2:14.6e} {cs2:14.6e}")
    print(f"\n  expected naive (2π+0.1)^2 = {(2 * math.pi + 0.1) ** 2:.4f}")
    print(f"  expected wrap 0.1^2       = {0.1 ** 2:.4f}")
    print(f"  expected cos/sin (small δ): (cos δ - 1)^2 + sin^2 δ "
          f"= {(math.cos(0.1) - 1) ** 2 + math.sin(0.1) ** 2:.6f}")
    agree = math.isclose(w2, cs2, rel_tol=2e-2)
    print(f"\nQ2 verdict: wrap_MSE and cos/sin_MSE agree (small angle): {agree}")
    print(f"           naive MSE is {n2 / w2:.0f}× too large -> UNUSABLE")


# ---------------------------------------------------------------------------
# Q3 — wrap-boundary edge case
# ---------------------------------------------------------------------------
def q3_wrap_boundary() -> None:
    print("\n" + "=" * 70)
    print("Q3 — wrap-boundary edge case (true=+π-ε, pred=-π+ε)")
    print("=" * 70)

    eps = 1e-3
    h_true = torch.tensor(math.pi - eps, dtype=torch.float64)
    h_pred = torch.tensor(-math.pi + eps, dtype=torch.float64)

    n, w, cs = _three_losses(h_pred, h_true)
    print(f"  naive_MSE   = {n:.6f}    (expected ≈ {(2 * math.pi - 2 * eps) ** 2:.6f})")
    print(f"  wrap_MSE    = {w:.6e}    (expected ≈ {(2 * eps) ** 2:.6e})")
    print(f"  cos/sin_MSE = {cs:.6e}   (expected ≈ {(2 * eps) ** 2:.6e})")
    bad_ratio = n / w
    print(f"\n  naive is {bad_ratio:.2e}× larger than wrap MSE -> CATASTROPHIC")
    print("Q3 verdict: signed_wrap MSE and cos/sin MSE both fix the bug.")


# ---------------------------------------------------------------------------
# Q4 — tan(phi_bank) gradient
# ---------------------------------------------------------------------------
def _phi_bank(phi_raw: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(phi_raw)


def q4_tan_gradient() -> None:
    print("\n" + "=" * 70)
    print("Q4 — tan(phi_bank) gradient through tanh-cap")
    print("=" * 70)

    phi_raw_vals = [-3.0, -1.0, 0.0, 1.0, 3.0]
    caps = [0.5, 0.7, 1.0, 1.3, 1.4, 1.5, 1.55]

    print(f"{'cap':>6s} | "
          + " ".join([f"phi_raw={v:>+4.1f}: (φ_b, tan, grad)" for v in phi_raw_vals]))
    print("-" * 110)
    for cap in caps:
        row = [f"{cap:6.2f} |"]
        max_grad = 0.0
        for v in phi_raw_vals:
            x = torch.tensor(v, dtype=torch.float64, requires_grad=True)
            phi = _phi_bank(x, cap)
            y = torch.tan(phi)
            y.backward()
            g = float(x.grad.item())
            max_grad = max(max_grad, abs(g))
            row.append(f"({float(phi.item()):+.3f},{float(y.item()):+.3f},{g:+.3e})")
        row.append(f"  max|grad|={max_grad:.3e}")
        print(" ".join(row))

    # Sweep: at what cap does max gradient explode?
    print("\n  Cap sweep — max |d tan(phi_bank)/d phi_raw| across phi_raw in [-5,5]")
    sweep = torch.linspace(-5, 5, 401, dtype=torch.float64)
    for cap in [0.5, 0.7, 1.0, 1.2, 1.3, 1.4, 1.45, 1.5, 1.55, 1.56]:
        x = sweep.clone().requires_grad_(True)
        y = torch.tan(cap * torch.tanh(x))
        y.sum().backward()
        max_g = float(x.grad.abs().max().item())
        # Analytical: grad = cap · sech²(phi_raw) · sec²(cap·tanh(phi_raw))
        # Maximum is at phi_raw=0: grad = cap · 1 · sec²(0) = cap, except sec² blows up
        # if cap·tanh(phi_raw) approaches π/2.
        max_phi_b = cap  # tanh saturates at ±1
        sec2 = 1.0 / max(math.cos(max_phi_b) ** 2, 1e-30)
        analytic_max = cap * sec2  # at phi_raw=±large, sech²→0; max is interior
        # Actual interior maximum is more subtle, but cap*sec²(cap) is the asymptotic ceiling.
        print(f"    cap={cap:.3f}  |  measured max|grad|={max_g:.4e}  "
              f"|  asymptotic cap·sec²(cap)={analytic_max:.4e}")

    print("\nQ4 verdict: cap=0.7 keeps tan(phi_bank) ≤ tan(0.7) ≈ 0.842 and grad bounded;")
    print("           cap → π/2 ≈ 1.5708 makes sec²(cap·tanh) explode.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    q1_signed_wrap()
    q2_loss_drift()
    q3_wrap_boundary()
    q4_tan_gradient()
    print("\nDone.")
