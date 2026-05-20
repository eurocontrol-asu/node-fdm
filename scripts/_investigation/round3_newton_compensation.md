# Round 3 — does the Newton v2 NN backbone leak the same way as v3?

Falsification loop, round 3. Round 2 found that in `full_hybrid_v3`
(CL mass-aware), the NN backbone removes 99.97% of the frozen-NN
penalty under m_0 x 1.3 — an architecture leak through the rollout
feedback loop, despite AC4 (backbone never sees `m` as input) being
syntactically preserved. Round 3 measures whether `full_hybrid_v2`
(Newton formulation, `L = lift_residual_norm * m_ref * g + m * g`)
leaks the SAME way.

**H3a (the hypothesis to falsify)** — Newton v2 also exhibits
near-total NN compensation: live-NN recovery >= 95% (matching v3's
99.97%). If true, v3 is just a more efficient version of a leak both
formulations share; rolling back to v2 saves nothing.

**Falsifier** — H3a dies if recovery < 80% (Cas B: v2 leaks
materially less; rollback worth considering — or Cas C: v2
essentially clean, < 40%). The 80-95% band is ambiguous.

## Setup

- model: `full_hybrid_v2` (arch `node_adsb_hybrid_v2`, Newton)
- `val_limit` = 2000, device = `cpu`, m_0 factor = 1.3
- val samples rolled out: 2207, rollout steps: 60
- rollout = faithful copy of `ODETrainer._compute_batch_loss`
  (60-step euler, ClampedEuler integrator) — reused from round 2
- NN outputs captured by wrapping the `data_ode_long` layer at
  runtime (no source file modified)
- NN output channels (Newton): `fdm_t_minus_d_norm`, `fdm_lift_residual_norm`
  (vs v3's `fdm_t_minus_d_norm`, `fdm_cl_residual`)
- PhysicsLayer auto-branches on which lift channel is present (`layers/physics.py` §L179-235)
- loss-to-target = trainer trajectory MSE (`residual / norm_std`,
  alpha-weighted, MSE reduction)

## 1. NN-output RMS *relative* divergence (baseline vs m_0x1.3)

Per channel, per rollout step:
`sqrt(mean_N((pert - base)^2)) / sqrt(mean_N(base^2))`.

### `fdm_t_minus_d_norm`

(captured 59 NN-output steps for a 60-state
rollout — euler does seq_len-1 dynamics evaluations per batch.)

| step | RMS rel. divergence | abs. divergence | baseline RMS magnitude |
|-----:|--------------------:|----------------:|-----------------------:|
| 1 | 18.36% | 0.0770379 | 0.419487 |
| 10 | 16.23% | 0.0670539 | 0.413187 |
| 30 | 27.1% | 0.104185 | 0.38449 |

Peak RMS relative divergence: **53.62%** (step 42).

### `fdm_lift_residual_norm`

(captured 59 NN-output steps for a 60-state
rollout — euler does seq_len-1 dynamics evaluations per batch.)

| step | RMS rel. divergence | abs. divergence | baseline RMS magnitude |
|-----:|--------------------:|----------------:|-----------------------:|
| 1 | 20.2% | 0.00248725 | 0.0123112 |
| 10 | 41.96% | 0.00274263 | 0.00653668 |
| 30 | 34.51% | 0.00272808 | 0.0079057 |

Peak RMS relative divergence: **56.41%** (step 42).

## 2. Three-way loss-to-target comparison (the counterfactual)

Counterfactual question — *is the NN-output divergence loss-reducing?*

* **baseline** — m_0 unperturbed, NN live.
* **perturbed, NN live** — m_0 x 1.3, NN free to react across steps.
* **perturbed, NN frozen** — m_0 x 1.3, but `data_ode_long` outputs
  frozen to their step-matched baseline values; the analytical
  PhysicsLayer still runs with the perturbed mass (Newton branch).

| rollout | loss-to-target | Δ vs baseline | Δ vs baseline (rel) |
|---------|---------------:|--------------:|--------------------:|
| baseline | 0.0212729 | 0 | 0% |
| perturbed, NN live | 0.0220198 | +0.000747 | +3.511% |
| perturbed, NN frozen | 0.0609875 | +0.0397 | +186.7% |
| _sanity: frozen at m_0x1.0_ | 0.0212729 | +0 | +0% |

_Sanity check_ — replaying the baseline NN outputs at m_0x1.0 (unperturbed mass) must reproduce the baseline loss exactly. Any drift would indicate a broken step counter / replay wiring.

**Frozen-NN penalty** (perturbed-frozen - baseline): `+0.03971` (+186.7%).
**Live-NN penalty** (perturbed-live - baseline): `+0.0007469` (+3.511%).
**Recovery ratio** `1 - (live_penalty / frozen_penalty)` = **98.12%**.

## 3. Loss-curvature sweep over m_0 factor

`identifiability_test`-style val loss at a range of m_0 factors.

| m_0 factor | val loss-to-target | Δ vs factor 1.0 (rel) |
|-----------:|-------------------:|----------------------:|
| 0.7 | 0.0267439 | +25.72% |
| 0.85 | 0.0227605 | +6.993% |
| 1 | 0.0212729 | +0% |
| 1.15 | 0.021495 | +1.044% |
| 1.3 | 0.0220198 | +3.511% |

Sweep loss span (max-min)/loss@1.0 = **25.72%**. Argmin at factor **1**.

## 4. Side-by-side: v2 (Newton, this round) vs v3 (CL, round 2)

| metric | v2 (Newton) | v3 (CL) |
|---|---:|---:|
| baseline loss | 0.0212729 | 0.0226051 |
| perturbed-live loss | 0.0220198 | 0.0226223 |
| perturbed-frozen loss | 0.0609875 | 0.0767264 |
| live penalty (% baseline) | +3.511% | +0.07613% |
| frozen penalty (% baseline) | +186.7% | +239.4% |
| recovery ratio | **98.12%** | **99.97%** |
| NN-out peak div. (t_minus_d_norm) | 53.62% | 51.23% |
| NN-out peak div. (lift channel) | 56.41% *(lift_residual_norm)* | 70.78% *(cl_residual)* |

Note: the lift-channel rows compare DIFFERENT physical quantities (v2's `lift_residual_norm` and v3's `cl_residual`); they are listed side-by-side because each is the lift-NN-output of its respective PhysicsLayer branch, not because they are directly commensurable.

## Verdict

Recovery ratio = **98.12%**. Classification: **Cas A**.

v2 removes 98.12% of the frozen-NN penalty, matching v3's 99.97%. The Newton formulation leaks the SAME way: the rollout feedback loop reads the diverged state and feeds compensating NN outputs into the analytical PhysicsLayer. v3 is not uniquely pathological — it's just the same leak with a different lift reconstruction. Rolling back to v2 saves nothing; the fix must repair the MassEncoder coupling itself.

## VERDICT: Cas A — both formulations leak equally
