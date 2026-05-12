# Hyperparameter sweep — runner & matrix

Orchestrates `fdm train -> fdm predict -> fdm evaluate` across a sweep-by-axis
matrix (one axis varies at a time around a fixed baseline) on 1–2 GPUs.

## Quick start

```bash
# 1. Inspect the matrix (no I/O, no GPU)
uv run python -m scripts.sweep_runner plan --smoke

# 2. Smoke test: 8 runs, train_limit=50, epochs=2 — verifies wiring end-to-end
uv run python -m scripts.sweep_runner run \
    --smoke \
    --config-template config.yaml \
    --gpus 0,1 --per-gpu 2

# 3. Inspect progress / failures
uv run python -m scripts.sweep_runner status --results-dir results_smoke

# 4. Aggregate per-run metrics into a single parquet
uv run python -m scripts.sweep_runner aggregate --results-dir results_smoke
```

For the full sweep, drop `--smoke`. Results land in `results/` by default.

## What each run produces

```
results_smoke/{run_id}/
├── config.yaml         # generated from template + per-run training overrides
├── run_config.json     # the full RunConfig (Pydantic)
├── train.log           # fdm train stdout/stderr
├── predict.log
├── evaluate.log
├── status.json         # {status, exit_codes, wall_clock_sec, gpu_id}
└── metrics.json        # parsed performance.parquet (score_primary + breakdown)
```

`queue.json` at the results root tracks pending / running / completed / failed
across invocations, so rerunning resumes where the previous call stopped.

## Matrix structure (default, ~30 runs)

* **Baseline**: `bs=128, epochs=50, lr=1e-3, weighting=on, alpha=0.5, activation=silu`
* **Axes** (one varies at a time):
  * `batch_size ∈ {32, 64, 256}`
  * `epochs ∈ {100}`
  * `lr ∈ {5e-4, 3e-3}`
  * `weighting ∈ {off}` (alpha then ignored)
  * `alpha ∈ {0.3, 0.7}` (weighting forced on)
  * `activation ∈ {relu}`
* **Seeds**: `{0, 1, 2}` per unique config

`scripts/sweep_matrix.py` is the authoritative spec — edit `Baseline` /
`SweepAxes` there to change the study.

## Key CLI flags exposed (this branch)

The runner relies on three flags that were added in the same change as this
script:

* `fdm train --activation {silu,relu,gelu,tanh}` — hidden-layer activation
* `fdm train --seed <int>` — seeds torch / numpy / random + DataLoader generator
* `fdm train --mode-weight-alpha <float>` — override `cfg.training.mode_weight_alpha`
* `fdm evaluate --model-name <run_id>` — read predictions from
  `<predict_dir>/<run_id>/` and write metrics to
  `<data_dir>/model_performance/<run_id>/`

Without these, every run would overwrite a shared `performance.parquet` and
seeds would be no-ops.

## Smoke vs full

|                | smoke           | full              |
|----------------|-----------------|-------------------|
| `train_limit`  | 50              | 5000              |
| `epochs`       | 2               | 50 / 100          |
| `seeds`        | `{0, 1}`        | `{0, 1, 2}`       |
| total runs     | ~8              | ~30               |
| per-run cost   | ~1–3 min on GPU | ~30 min – 2 h     |
| use            | wiring check    | hyperparam study  |

## Resuming a crashed run

Just rerun the same `run` command. Anything in `completed` is skipped;
anything left in `running` from a crashed parent is requeued.

## GPU concurrency

* `--gpus 0,1` runs on both GPUs.
* `--per-gpu 3` co-schedules three training instances per GPU. Adjust down if
  you see OOM in `train.log` (no automatic memory probing yet — start with
  `--per-gpu 1` on the full matrix to calibrate VRAM headroom, then ramp up).
* `--gpus cpu` disables CUDA entirely (useful for debugging on a laptop).
