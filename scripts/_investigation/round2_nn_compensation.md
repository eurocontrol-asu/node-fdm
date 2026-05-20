# Round 2 — does the v3 NN backbone compensate the m_0 perturbation?

Falsification loop, round 2. Round 1 found that perturbing the
MassEncoder-predicted `m_0` by 1.3x moves the predicted state
trajectory by ~7*sigma_obs in altitude over 60 steps, yet the val
loss barely moves. Round 2 explains WHY the loss stays flat.

**H2b (the hypothesis to falsify)** — the v3 NN backbone actively
compensates the m_0 perturbation: through the rollout feedback loop
it sees the diverged state and emits different `cl_residual` /
`t_minus_d_norm` that pull the trajectory back toward the observed
target, keeping the loss flat (an architecture leak).

**H2a (the null)** — the ADS-B val data genuinely does not constrain
`m_0`; the loss landscape is intrinsically flat in `m_0`. The NN
outputs drift only slightly (passive reaction to a modestly shifted
input distribution), not in a loss-reducing way.

**Falsifier** — H2b dies if the RMS *relative* divergence of each NN
output channel stays below ~10% at every
sampled rollout step. H2b survives only if NN outputs diverge
strongly AND that divergence demonstrably reduces loss-to-target vs
the frozen-NN counterfactual.

## Setup

- model: `full_hybrid_v3` (arch `node_adsb_hybrid_v3`, CL mass-aware)
- `val_limit` = 2000, device = `cpu`, m_0 factor = 1.3
- val samples rolled out: 2207, rollout steps: 60
- rollout = faithful copy of `ODETrainer._compute_batch_loss`
  (60-step euler, ClampedEuler integrator) — reused from round 1
- NN outputs captured by wrapping the `data_ode_long` layer at
  runtime (no source file modified)
- loss-to-target = trainer trajectory MSE (`residual / norm_std`,
  alpha-weighted, MSE reduction)

## 1. NN-output RMS *relative* divergence (baseline vs m_0x1.3)

Per channel, per rollout step:
`sqrt(mean_N((pert - base)^2)) / sqrt(mean_N(base^2))`.
Falsifier threshold = 10%.

### `fdm_t_minus_d_norm`

(captured 59 NN-output steps for a 60-state
rollout — euler does seq_len-1 dynamics evaluations per batch.)

| step | RMS rel. divergence | abs. divergence | baseline RMS magnitude |
|-----:|--------------------:|----------------:|-----------------------:|
| 1 | 14.93% | 0.0676566 | 0.453126 |
| 10 | 18.17% | 0.0712894 | 0.392341 |
| 30 | 27.41% | 0.102494 | 0.373903 |

Peak RMS relative divergence: **51.23%** (step 36).

### `fdm_cl_residual`

(captured 59 NN-output steps for a 60-state
rollout — euler does seq_len-1 dynamics evaluations per batch.)

| step | RMS rel. divergence | abs. divergence | baseline RMS magnitude |
|-----:|--------------------:|----------------:|-----------------------:|
| 1 | 20.19% | 0.0020864 | 0.0103363 |
| 10 | 47.05% | 0.00208313 | 0.00442795 |
| 30 | 53.86% | 0.00209461 | 0.00388909 |

Peak RMS relative divergence: **70.78%** (step 42).

## 2. Three-way loss-to-target comparison (the counterfactual)

Counterfactual question — *is the NN-output divergence loss-reducing?*

* **baseline** — m_0 unperturbed, NN live.
* **perturbed, NN live** — m_0 x 1.3, NN free to react across steps.
* **perturbed, NN frozen** — m_0 x 1.3, but `data_ode_long` outputs
  frozen to their step-matched baseline values; the analytical
  PhysicsLayer still runs with the perturbed mass.

If *perturbed-live* loss is much lower than *perturbed-frozen* -> the
NN IS actively compensating -> H2b. If the two are about equal -> the
NN is not compensating -> H2a.

| rollout | loss-to-target | Δ vs baseline | Δ vs baseline (rel) |
|---------|---------------:|--------------:|--------------------:|
| baseline | 0.0226051 | 0 | 0% |
| perturbed, NN live | 0.0226223 | +1.72e-05 | +0.07613% |
| perturbed, NN frozen | 0.0767264 | +0.0541 | +239.4% |
| _sanity: frozen at m_0x1.0_ | 0.0226051 | +0 | +0% |

_Sanity check_ — replaying the baseline NN outputs at m_0x1.0 (unperturbed mass) must reproduce the baseline loss exactly. Any drift would indicate a broken step counter / replay wiring.

**Frozen-NN penalty** (perturbed-frozen - baseline): `+0.05412` (+239.4%).
**Live-NN penalty** (perturbed-live - baseline): `+1.721e-05` (+0.07613%).
**Fraction of the frozen penalty removed by the live NN**: `99.97%`.

Reading: a large positive *removed* fraction (live NN erases the
frozen penalty) => the NN compensates => H2b. A *removed* fraction
near 0% (or negative) => the NN does not compensate; the frozen
and live perturbed losses move together => H2a.

## 3. Loss-curvature sweep over m_0 factor

`identifiability_test`-style val loss at a range of m_0 factors.
Flat across the whole sweep => data genuinely uninformative about
`m_0` (supports H2a). A visible minimum near factor 1.0 => the loss
does carry `m_0` information (the 2-point 1.3x test was too coarse).

| m_0 factor | val loss-to-target | Δ vs factor 1.0 (rel) |
|-----------:|-------------------:|----------------------:|
| 0.7 | 0.0290068 | +28.32% |
| 0.85 | 0.0247085 | +9.305% |
| 1 | 0.0226051 | +0% |
| 1.15 | 0.0222518 | -1.563% |
| 1.3 | 0.0226223 | +0.07613% |

Sweep loss span (max-min)/loss@1.0 = **29.88%**. Argmin at factor **1.15**.

## Verdict

NN outputs diverge strongly (fdm_t_minus_d_norm, fdm_cl_residual exceed the 10% threshold; overall peak 70.8%) AND that divergence is loss-reducing: the live NN removes 100% of the frozen-NN penalty. The backbone actively re-steers the trajectory back toward target — an architecture leak. Repair the MassEncoder coupling before Phase 2.

## VERDICT: H2b SURVIVES — NN compensates
