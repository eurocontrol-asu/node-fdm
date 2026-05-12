"""Sweep orchestrator: train -> predict -> evaluate per RunConfig, multi-GPU queue.

Usage (run from repo root)::

    uv run python -m scripts.sweep_runner plan --smoke
    uv run python -m scripts.sweep_runner run --smoke --config-template config.yaml
    uv run python -m scripts.sweep_runner status --results-dir results_smoke
    uv run python -m scripts.sweep_runner aggregate --results-dir results_smoke

State lives in ``{results_dir}/queue.json`` (idempotent: rerunning resumes).
Each run has its own ``{results_dir}/{run_id}/`` directory containing the
overridden YAML config, train/predict/evaluate logs, and ``metrics.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import structlog

from scripts.sweep_matrix import RunConfig, default_matrix, smoke_matrix

__all__ = ["app", "main"]

log = structlog.get_logger("sweep_runner")

app = cyclopts.App(
    name="sweep_runner",
    help="Orchestrate a hyperparameter sweep over fdm train -> predict -> evaluate.",
)


# ---------------------------------------------------------------------------
# Queue state persistence
# ---------------------------------------------------------------------------


@dataclass
class QueueState:
    """Mutable queue snapshot persisted to disk between runs."""

    pending: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "pending": list(self.pending),
            "running": list(self.running),
            "completed": list(self.completed),
            "failed": list(self.failed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> QueueState:
        return cls(
            pending=list(data.get("pending", [])),
            running=list(data.get("running", [])),
            completed=list(data.get("completed", [])),
            failed=list(data.get("failed", [])),
        )


def _queue_path(results_dir: Path) -> Path:
    return results_dir / "queue.json"


def _load_queue(results_dir: Path) -> QueueState:
    path = _queue_path(results_dir)
    if not path.exists():
        return QueueState()
    return QueueState.from_dict(json.loads(path.read_text()))


def _save_queue(results_dir: Path, state: QueueState) -> None:
    _queue_path(results_dir).write_text(json.dumps(state.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Config YAML generation per run
# ---------------------------------------------------------------------------


def _render_run_config(template_path: Path, run: RunConfig) -> str:
    """Inline the per-run training overrides into a YAML config copy.

    The pipeline ``PipelineConfig`` only exposes ``use_mode_weights`` and
    ``mode_weight_alpha`` under ``training:``; ``batch_size``, ``epochs``,
    ``lr``, ``activation`` and ``seed`` are passed as CLI flags. This
    helper keeps the YAML in sync with the CLI overrides so the train
    output captures the full state of the run.
    """
    import yaml

    cfg: dict[str, Any] = yaml.safe_load(template_path.read_text()) or {}
    training = dict(cfg.get("training", {}))
    training["use_mode_weights"] = run.use_mode_weights
    training["mode_weight_alpha"] = run.mode_weight_alpha
    cfg["training"] = training
    return yaml.safe_dump(cfg, sort_keys=False)


# ---------------------------------------------------------------------------
# Subprocess wrappers
# ---------------------------------------------------------------------------


def _run_subprocess(
    args: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    timeout_s: int | None = None,
) -> int:
    """Run ``args``, tee stdout+stderr to ``log_path``, return exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as fh:
        fh.write(f"\n$ {' '.join(args)}\n".encode())
        proc = subprocess.Popen(  # noqa: S603  # args is a fixed argv list, no shell
            args,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return 124  # convention: timeout exit code


def _train_args(run: RunConfig, config_path: Path) -> list[str]:
    args = [
        "uv",
        "run",
        "fdm",
        "train",
        "--arch",
        run.arch,
        "--config",
        str(config_path),
        "--model-name",
        run.run_id,
        "--epochs",
        str(run.epochs),
        "--batch-size",
        str(run.batch_size),
        "--lr",
        str(run.lr),
        "--method",
        run.method,
        "--seq-len",
        str(run.seq_len),
        "--train-limit",
        str(run.train_limit),
        "--activation",
        run.activation,
        "--seed",
        str(run.seed),
        "--device",
        os.environ.get("FDM_DEVICE", "cuda"),
    ]
    if run.use_mode_weights:
        args.append("--use-mode-weights")
        args.extend(["--mode-weight-alpha", str(run.mode_weight_alpha)])
    else:
        args.append("--no-use-mode-weights")
    return args


def _predict_args(run: RunConfig, config_path: Path) -> list[str]:
    args = [
        "uv",
        "run",
        "fdm",
        "predict",
        "--arch",
        run.arch,
        "--config",
        str(config_path),
        "--local-model",
        "--model-name",
        run.run_id,
        "--device",
        os.environ.get("FDM_DEVICE", "cuda"),
    ]
    if run.predict_limit is not None:
        args.extend(["--limit", str(run.predict_limit)])
    return args


def _evaluate_args(run: RunConfig, config_path: Path) -> list[str]:
    return [
        "uv",
        "run",
        "fdm",
        "evaluate",
        "--arch",
        run.arch,
        "--config",
        str(config_path),
        "--model-name",
        run.run_id,
    ]


# ---------------------------------------------------------------------------
# Per-run executor — invoked inside a worker process
# ---------------------------------------------------------------------------


def _execute_run(
    run_dict: dict[str, Any],
    template_path: str,
    results_dir: str,
    gpu_id: int,
    timeout_s: int | None,
) -> dict[str, Any]:
    """Pipeline a single run: write YAML, train, predict, evaluate, parse metrics.

    Runs in a child process so each run has its own ``CUDA_VISIBLE_DEVICES``
    binding without touching the parent process env.
    """
    run = RunConfig.model_validate(run_dict)
    template = Path(template_path)
    out_dir = Path(results_dir) / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.yaml"
    config_path.write_text(_render_run_config(template, run))
    (out_dir / "run_config.json").write_text(run.model_dump_json(indent=2))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    # Each run gets its own torch.compile cache to avoid contention between
    # concurrent training processes hitting ~/.cache/torch/inductor/.
    cache_dir = out_dir / "torch_cache"
    cache_dir.mkdir(exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    env["TRITON_CACHE_DIR"] = str(cache_dir / "triton")

    status: dict[str, Any] = {
        "run_id": run.run_id,
        "gpu_id": gpu_id,
        "stage": "train",
        "exit_codes": {},
        "started_at": time.time(),
    }

    for stage, args in (
        ("train", _train_args(run, config_path)),
        ("predict", _predict_args(run, config_path)),
        ("evaluate", _evaluate_args(run, config_path)),
    ):
        status["stage"] = stage
        rc = _run_subprocess(
            args,
            log_path=out_dir / f"{stage}.log",
            env=env,
            timeout_s=timeout_s,
        )
        status["exit_codes"][stage] = rc
        if rc != 0:
            status["status"] = "failed"
            status["finished_at"] = time.time()
            (out_dir / "status.json").write_text(json.dumps(status, indent=2))
            return status

    metrics = _parse_metrics(out_dir, run, template)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    status["status"] = "completed"
    status["finished_at"] = time.time()
    status["wall_clock_sec"] = status["finished_at"] - status["started_at"]
    (out_dir / "status.json").write_text(json.dumps(status, indent=2))
    return status


def _parse_metrics(out_dir: Path, run: RunConfig, template_path: Path) -> dict[str, Any]:
    """Read ``performance.parquet`` from the typecode's run and extract metrics.

    Schema is flat::

        {
            "run_id": ...,
            "score_primary": MAE(raw_alt_m, All phases, pred_),
            "score_breakdown": {variable: {phase: {MAE, MAPE, ME}}},
            "bada_baseline":  same shape, Model=bada_,
        }
    """
    import polars as pl
    import yaml

    cfg = yaml.safe_load(template_path.read_text()) or {}
    data_dir = Path(cfg["paths"]["data_dir"])
    perf_path = data_dir / "model_performance" / run.run_id / "performance.parquet"

    metrics: dict[str, Any] = {"run_id": run.run_id, "score_primary": None}
    if not perf_path.exists():
        metrics["error"] = f"performance.parquet missing: {perf_path}"
        return metrics

    df = pl.read_parquet(perf_path)

    def _extract(model_prefix: str) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for row in df.filter(pl.col("Model") == model_prefix).iter_rows(named=True):
            var = row["Variable"]
            phase = row["Phase"]
            out.setdefault(var, {})[phase] = {
                "MAE": float(row["MAE"]),
                "MAPE": float(row.get("MAPE (%)", 0.0)),
                "ME": float(row["ME"]),
            }
        return out

    pred = _extract("PRED")
    bada = _extract("BADA")

    metrics["score_breakdown"] = pred
    metrics["bada_baseline"] = bada

    score_keys = {
        "mae_alt": ("Altitude [m]", "raw_alt_m"),
        "mae_tas": ("True airspeed [m/s]", "era_tas_ms"),
        "mae_gamma": ("Flight path angle [deg]", "fdm_gamma_rad"),
        "mae_heading": ("Heading [deg]", "fdm_heading_rad"),
    }
    for key, (label, fallback) in score_keys.items():
        entry = pred.get(label, {}).get("All phases")
        if entry is None:
            entry = pred.get(fallback, {}).get("All phases")
        metrics[key] = entry["MAE"] if entry is not None else None

    metrics["score_primary"] = metrics["mae_alt"]
    metrics["perf_path"] = str(perf_path)
    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class SlotPool:
    """Concurrent-slot allocator.

    Each ``slot_id`` is unique even when several slots map to the same GPU
    (oversubscription via ``--per-gpu``). ``slot_to_gpu[slot_id]`` gives the
    actual CUDA device id to inject into ``CUDA_VISIBLE_DEVICES``.
    """

    slot_to_gpu: dict[int, int]
    busy: dict[int, str] = field(default_factory=dict)

    def acquire(self) -> int | None:
        for slot_id in self.slot_to_gpu:
            if slot_id not in self.busy:
                return slot_id
        return None

    def release(self, slot_id: int) -> None:
        self.busy.pop(slot_id, None)

    def gpu_of(self, slot_id: int) -> int:
        return self.slot_to_gpu[slot_id]


def _build_slots(gpus: list[int], per_gpu: int) -> dict[int, int]:
    """Assign a unique ``slot_id`` per concurrent worker, mapped to its GPU.

    Two slots on the same GPU must not collide on the busy table — the
    previous flat ``list[int]`` flattened both into the same key. Returns
    ``{slot_id: gpu_id}`` instead.
    """
    return dict(enumerate(gpu for gpu in gpus for _ in range(per_gpu)))


def _filter_runs(runs: list[RunConfig], state: QueueState) -> list[RunConfig]:
    """Drop runs already marked completed/failed in a previous invocation."""
    done = set(state.completed) | set(state.failed)
    return [r for r in runs if r.run_id not in done]


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def _matrix(*, smoke: bool, arch: str) -> list[RunConfig]:
    return smoke_matrix(arch=arch) if smoke else default_matrix(arch=arch)


@app.command
def plan(
    *,
    smoke: Annotated[
        bool,
        cyclopts.Parameter(help="Use the smoke matrix (capped train_limit, epochs, seeds)"),
    ] = False,
    arch: Annotated[str, cyclopts.Parameter(help="Architecture identifier")] = "adsb",
) -> None:
    """Print the matrix that would be executed; touches no files."""
    runs = _matrix(smoke=smoke, arch=arch)
    print(f"runs: {len(runs)}")
    print(
        f"{'run_id':<32} {'axis':<14} {'seed':<5} "
        f"{'bs':<4} {'epoch':<5} {'lr':<8} {'w':<2} {'alpha':<5} {'act':<5} {'lim':<5}"
    )
    for r in runs:
        print(
            f"{r.run_id:<32} {r.axis:<14} {r.seed:<5} "
            f"{r.batch_size:<4} {r.epochs:<5} {r.lr:<8g} "
            f"{int(r.use_mode_weights):<2} {r.mode_weight_alpha:<5g} "
            f"{r.activation:<5} {r.train_limit:<5}"
        )


@app.command
def run(
    *,
    smoke: Annotated[
        bool,
        cyclopts.Parameter(help="Use the smoke matrix (capped train_limit, epochs, seeds)"),
    ] = False,
    arch: Annotated[str, cyclopts.Parameter(help="Architecture identifier")] = "adsb",
    config_template: Annotated[
        Path,
        cyclopts.Parameter(
            name="--config-template",
            help="Base YAML config; overrides for use_mode_weights/alpha are inlined per run",
        ),
    ] = Path("config.yaml"),
    results_dir: Annotated[
        Path | None,
        cyclopts.Parameter(
            name="--results-dir",
            help="Output root (default: results_smoke/ in smoke mode, results/ otherwise)",
        ),
    ] = None,
    gpus: Annotated[
        str,
        cyclopts.Parameter(
            help="Comma-separated GPU IDs (e.g. '0,1' for two GPUs). Use 'cpu' to disable CUDA."
        ),
    ] = "0,1",
    per_gpu: Annotated[
        int,
        cyclopts.Parameter(
            name="--per-gpu",
            help="Concurrent training instances per GPU (memory permitting)",
        ),
    ] = 3,
    timeout_s: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--timeout-s",
            help="Per-run wall-clock limit. None disables (smoke default: 1800s).",
        ),
    ] = None,
) -> None:
    """Execute the sweep with a queue, resuming any completed runs from disk."""
    if results_dir is None:
        results_dir = Path("results_smoke") if smoke else Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if timeout_s is None and smoke:
        timeout_s = 1800  # 30 min cap per smoke run

    runs_all = _matrix(smoke=smoke, arch=arch)
    state = _load_queue(results_dir)
    state.running = []  # discard stale running entries from a prior crash
    todo = _filter_runs(runs_all, state)

    state.pending = [r.run_id for r in todo]
    _save_queue(results_dir, state)
    log.info(
        "sweep_start",
        total=len(runs_all),
        todo=len(todo),
        results_dir=str(results_dir),
    )

    if gpus.lower() == "cpu":
        slot_to_gpu = dict.fromkeys(range(per_gpu), -1)
    else:
        gpu_ids = [int(g) for g in gpus.split(",") if g.strip()]
        slot_to_gpu = _build_slots(gpu_ids, per_gpu)
    pool = SlotPool(slot_to_gpu=slot_to_gpu)

    futures: dict[Future[dict[str, Any]], tuple[str, int]] = {}
    runs_by_id = {r.run_id: r for r in todo}
    remaining = list(todo)

    with ProcessPoolExecutor(max_workers=len(slot_to_gpu)) as executor:
        while remaining or futures:
            while remaining:
                slot_id = pool.acquire()
                if slot_id is None:
                    break
                run_cfg = remaining.pop(0)
                pool.busy[slot_id] = run_cfg.run_id
                gpu_id = pool.gpu_of(slot_id)
                state.pending = [
                    rid for rid in state.pending if rid != run_cfg.run_id
                ]
                state.running.append(run_cfg.run_id)
                _save_queue(results_dir, state)
                log.info(
                    "run_dispatch",
                    run_id=run_cfg.run_id,
                    slot_id=slot_id,
                    gpu_id=gpu_id,
                )
                fut = executor.submit(
                    _execute_run,
                    run_cfg.model_dump(),
                    str(config_template),
                    str(results_dir),
                    gpu_id,
                    timeout_s,
                )
                futures[fut] = (run_cfg.run_id, slot_id)

            if not futures:
                break

            done_fut = next(_as_completed(futures))
            run_id, slot_id = futures.pop(done_fut)
            pool.release(slot_id)
            state.running = [rid for rid in state.running if rid != run_id]
            try:
                result = done_fut.result()
                if result.get("status") == "completed":
                    state.completed.append(run_id)
                    log.info(
                        "run_done",
                        run_id=run_id,
                        wall_clock_sec=result.get("wall_clock_sec"),
                    )
                else:
                    state.failed.append(run_id)
                    log.error(
                        "run_failed",
                        run_id=run_id,
                        stage=result.get("stage"),
                        exit_codes=result.get("exit_codes"),
                    )
            except Exception as exc:
                state.failed.append(run_id)
                log.error("run_crashed", run_id=run_id, error=str(exc))
            _save_queue(results_dir, state)

    log.info(
        "sweep_done",
        completed=len(state.completed),
        failed=len(state.failed),
        total=len(runs_all),
    )
    _ = runs_by_id  # quiet unused warning when no runs are dispatched


def _as_completed(futures: dict[Future[dict[str, Any]], Any]) -> Iterable[Future[dict[str, Any]]]:
    """Yield futures as they complete (poll-based, short sleep)."""
    while futures:
        for fut in list(futures):
            if fut.done():
                yield fut
                return
        time.sleep(0.5)


@app.command
def status(
    *,
    results_dir: Annotated[Path, cyclopts.Parameter(name="--results-dir")] = Path("results"),
) -> None:
    """Show queue snapshot: pending/running/completed/failed counts."""
    state = _load_queue(results_dir)
    print(f"results_dir: {results_dir}")
    print(f"  pending:   {len(state.pending)}")
    print(f"  running:   {len(state.running)}")
    print(f"  completed: {len(state.completed)}")
    print(f"  failed:    {len(state.failed)}")
    if state.failed:
        print("\nfailed runs:")
        for rid in state.failed:
            print(f"  - {rid}")


@app.command
def aggregate(
    *,
    results_dir: Annotated[Path, cyclopts.Parameter(name="--results-dir")] = Path("results"),
    output: Annotated[
        Path | None,
        cyclopts.Parameter(help="Output parquet path (default: {results_dir}/summary.parquet)"),
    ] = None,
) -> None:
    """Merge per-run ``metrics.json`` files into a single parquet for analysis."""
    import polars as pl

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m_path = run_dir / "metrics.json"
        rc_path = run_dir / "run_config.json"
        if not m_path.exists() or not rc_path.exists():
            continue
        metrics = json.loads(m_path.read_text())
        run_cfg = json.loads(rc_path.read_text())
        row = {
            **{
                k: run_cfg[k]
                for k in (
                    "run_id",
                    "axis",
                    "seed",
                    "batch_size",
                    "epochs",
                    "lr",
                    "use_mode_weights",
                    "mode_weight_alpha",
                    "activation",
                    "train_limit",
                )
            },
            "score_primary": metrics.get("score_primary"),
            "mae_alt": metrics.get("mae_alt"),
            "mae_tas": metrics.get("mae_tas"),
            "mae_gamma": metrics.get("mae_gamma"),
            "mae_heading": metrics.get("mae_heading"),
        }
        rows.append(row)

    if not rows:
        log.warning("aggregate_empty", results_dir=str(results_dir))
        return

    out = output or (results_dir / "summary.parquet")
    pl.DataFrame(rows).write_parquet(out)
    log.info("aggregate_done", rows=len(rows), output=str(out))


def main() -> None:
    """Entry point for ``python -m scripts.sweep_runner``."""
    app()


if __name__ == "__main__":
    main()
    sys.exit(0)
