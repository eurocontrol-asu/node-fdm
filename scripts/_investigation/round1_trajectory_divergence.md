# Round 1 — m_0 trajectory-divergence measurement

Falsification loop, round 1. **H1**: `m_0` is genuinely weakly
identifiable in `full_hybrid_v3` (CL mass-aware) — perturbing the
predicted initial mass by factor 1.3 displaces the *predicted state
trajectory* by < 1*sigma_obs per state dim at the end of the 60-step
rollout.

**Falsifier**: H1 dies if the final-step divergence
`||z_perturbed - z_baseline||` exceeds **2*sigma_obs** on any of
{altitude, gamma, TAS}. H1 survives if divergence stays **< 1*sigma_obs**
on all dims.

**Measured quantity** — prediction-vs-prediction:
`sqrt( mean_batch( (z_pred_perturbed[step] - z_pred_baseline[step])^2 ) )`
per state dim. This is NOT the loss-vs-target distance that
`trainer.identifiability_test` reports.

## Setup

- `val_limit` = 2000, device = `cpu`, m_0 factor = 1.3
- Rollout = faithful copy of `ODETrainer._compute_batch_loss` (60-step euler, ClampedEuler integrator)
- sigma_obs (ADS-B, PHASE_1.5 doc s6/s10): altitude = 5 m, TAS = 1 m/s, gamma = 0.001 rad

## full_hybrid_v3 — CL mass-aware (arch node_adsb_hybrid_v3) — hypothesis target

- architecture: `node_adsb_hybrid_v3`
- MassEncoder present: `True`
- val samples rolled out: 2207, rollout steps: 60
- baseline mean m_0: 6.316e+04 (note: x_cols mass channel; see caveat below)
- realised RMS m_0 shift (perturbed - baseline): 1.211e+04

### Divergence curve — RMS(z_pert - z_base) at sampled steps

| step | altitude (m) | gamma (rad) | TAS (m/s) |
|-----:|-------------:|------------:|----------:|
| 1 | 0 | 0.00110746 | 0.432994 |
| 10 | 4.57543 | 0.00114683 | 1.98647 |
| 30 | 18.2433 | 0.00131909 | 3.89976 |
| 59 | 35.7166 | 0.00144962 | 4.36841 |

### Final-step (step 59) divergence in units of sigma_obs

| dim | divergence | sigma_obs | divergence / sigma_obs |
|-----|-----------:|----------:|-----------------------:|
| altitude | 35.7166 | 5 | 7.143 |
| gamma | 0.00144962 | 0.001 | 1.45 |
| TAS | 4.36841 | 1 | 4.368 |

**full_hybrid_v3 verdict**: `DIVERGES (>2 sigma_obs)` (max ratio = 7.143).

## full_hybrid_v2 — Newton baseline (arch node_adsb_hybrid_v2) — CONTROL

- architecture: `node_adsb_hybrid_v2`
- MassEncoder present: `True`
- val samples rolled out: 2207, rollout steps: 60
- baseline mean m_0: 6.486e+04 (note: x_cols mass channel; see caveat below)
- realised RMS m_0 shift (perturbed - baseline): 1.165e+04

### Divergence curve — RMS(z_pert - z_base) at sampled steps

| step | altitude (m) | gamma (rad) | TAS (m/s) |
|-----:|-------------:|------------:|----------:|
| 1 | 0 | 0.000814774 | 0.376171 |
| 10 | 4.31585 | 0.000931626 | 1.82648 |
| 30 | 15.3022 | 0.00118008 | 3.27334 |
| 59 | 27.6265 | 0.00159225 | 3.8574 |

### Final-step (step 59) divergence in units of sigma_obs

| dim | divergence | sigma_obs | divergence / sigma_obs |
|-----|-----------:|----------:|-----------------------:|
| altitude | 27.6265 | 5 | 5.525 |
| gamma | 0.00159225 | 0.001 | 1.592 |
| TAS | 3.8574 | 1 | 3.857 |

**full_hybrid_v2 verdict**: `DIVERGES (>2 sigma_obs)` (max ratio = 5.525).

## Cross-model verdict

| model | role | final-step max ratio | classification |
|-------|------|---------------------:|----------------|
| full_hybrid_v3 | CL mass-aware (target) | 7.143 | DIVERGES (>2 sigma_obs) |
| full_hybrid_v2 | Newton (control) | 5.525 | DIVERGES (>2 sigma_obs) |

The Newton control (v2) is expected to be identifiable — a large divergence there confirms the measurement actually detects an m_0 effect when one exists.

### Caveat — x_cols mass channel

`fdm_mass_kg` in `x_seq` is zero-padded in the val loader (stats_dict shows mean/std ~ 0). The perturbation is applied to the MassEncoder-predicted `m_0` injected into `x0` (trainer.py L601-604) — this matches `identifiability_test` exactly. The `baseline mean m_0` row reflects the x_cols channel, not the encoder output magnitude; the `realised RMS m_0 shift` row is the meaningful perturbation size.

## VERDICT: H1 DEAD — H2 confirmed

v3 final-step divergence exceeds 2*sigma_obs on at least one dim — m_0 DOES move the predicted trajectory substantially. The flat val_loss masked a real trajectory effect (sideways shift keeping MSE-to-target constant).
