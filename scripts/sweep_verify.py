"""End-to-end verification of a sweep run.

Usage::

    uv run python -m scripts.sweep_verify --results-dir results_smoke

Runs six checks in order, prints a ``PASS/WARN/FAIL`` line per check and
exits non-zero if any FAIL is found:

1. queue completeness         — queue.json says everything finished
2. per-run artifacts          — config / metrics / status / 3 logs present
3. namespace isolation        — model_performance / predicted_flights / models
                                have one sub-dir per run_id, no shared file
4. performance.parquet shape  — non-empty, expected columns, both pred_/bada_
5. score_primary distribution — non-null, finite, comparable across seeds
6. axis effect smoke summary  — mean score_primary grouped by axis

The script is read-only: it never writes anywhere except the existing
``results_dir`` (to refresh ``summary.parquet`` via ``aggregate``).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
import polars as pl
import yaml

__all__ = ["app", "main"]

app = cyclopts.App(
    name="sweep_verify",
    help="Run all sanity checks on a finished sweep in one command.",
)

Status = Literal["PASS", "WARN", "FAIL"]


def _emit(status: Status, name: str, msg: str = "") -> None:
    """Print a single check result line; coloured if running in a TTY."""
    colors = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}
    reset = "\033[0m"
    use_color = sys.stdout.isatty()
    tag = f"{colors[status]}{status:<4}{reset}" if use_color else f"{status:<4}"
    extra = f" — {msg}" if msg else ""
    print(f"  [{tag}] {name}{extra}")


def _check_queue(results_dir: Path) -> Status:
    queue_path = results_dir / "queue.json"
    if not queue_path.exists():
        _emit("FAIL", "queue.json present", str(queue_path))
        return "FAIL"
    state = json.loads(queue_path.read_text())
    pending = len(state.get("pending", []))
    running = len(state.get("running", []))
    completed = len(state.get("completed", []))
    failed = state.get("failed", [])
    summary = f"completed={completed}, failed={len(failed)}, pending={pending}, running={running}"
    if pending or running:
        _emit("WARN", "queue completeness", summary + " (sweep still in progress?)")
        return "WARN"
    if failed:
        _emit("FAIL", "queue completeness", summary + f" — failed: {failed}")
        return "FAIL"
    _emit("PASS", "queue completeness", summary)
    return "PASS"


def _check_artifacts(results_dir: Path) -> Status:
    expected = {"config.yaml", "run_config.json", "metrics.json", "status.json"}
    expected_logs = {"train.log", "predict.log", "evaluate.log"}
    bad: list[str] = []
    for run_dir in _run_dirs(results_dir):
        existing = {p.name for p in run_dir.iterdir()}
        missing = (expected | expected_logs) - existing
        if missing:
            bad.append(f"{run_dir.name}: missing {sorted(missing)}")
    if bad:
        for b in bad[:5]:
            _emit("FAIL", "per-run artifacts", b)
        if len(bad) > 5:
            _emit("FAIL", "per-run artifacts", f"...and {len(bad) - 5} more")
        return "FAIL"
    _emit("PASS", "per-run artifacts", f"{sum(1 for _ in _run_dirs(results_dir))} runs OK")
    return "PASS"


def _check_isolation(results_dir: Path, data_dir: Path) -> Status:
    """Each run must own its own sub-dir under model_performance / predicted_flights / models."""
    run_ids = {d.name for d in _run_dirs(results_dir)}
    if not run_ids:
        _emit("WARN", "namespace isolation", "no completed runs")
        return "WARN"

    problems: list[str] = []
    for sub in ("model_performance", "predicted_flights", "models"):
        root = data_dir / sub
        if not root.exists():
            problems.append(f"{root} missing")
            continue
        present = {d.name for d in root.iterdir() if d.is_dir()}
        not_isolated = run_ids - present
        if not_isolated:
            sample = next(iter(not_isolated))
            problems.append(f"{sub}: {len(not_isolated)} runs without own dir (e.g. {sample})")

    shared_perf = data_dir / "model_performance" / "performance.parquet"
    if shared_perf.exists():
        problems.append(f"shared file detected: {shared_perf} (concurrent writes risk)")

    if problems:
        for p in problems:
            _emit("FAIL", "namespace isolation", p)
        return "FAIL"
    _emit("PASS", "namespace isolation", f"{len(run_ids)} run_ids isolated across 3 sub-dirs")
    return "PASS"


def _check_perf_parquet(results_dir: Path, data_dir: Path) -> Status:
    """Sample a few performance.parquet files: shape, columns, both pred_/bada_."""
    expected_cols = {"Aircraft", "Variable", "Phase", "Model", "MAE", "Count"}
    sampled = 0
    bad: list[str] = []
    for run_dir in _run_dirs(results_dir):
        perf = data_dir / "model_performance" / run_dir.name / "performance.parquet"
        if not perf.exists():
            bad.append(f"{run_dir.name}: parquet missing")
            continue
        df = pl.read_parquet(perf)
        sampled += 1
        if df.is_empty():
            bad.append(f"{run_dir.name}: empty parquet")
            continue
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            bad.append(f"{run_dir.name}: missing cols {sorted(missing_cols)}")
        models = set(df["Model"].unique())
        if "pred_" not in models:
            bad.append(f"{run_dir.name}: no pred_ rows (predict step did not write any flight)")
        if "bada_" not in models:
            bad.append(f"{run_dir.name}: no bada_ rows (BADA absent — only matters if expected)")
    if bad:
        for b in bad[:5]:
            _emit("FAIL", "performance.parquet shape", b)
        if len(bad) > 5:
            _emit("FAIL", "performance.parquet shape", f"...and {len(bad) - 5} more")
        return "FAIL"
    _emit("PASS", "performance.parquet shape", f"{sampled} files OK (cols + pred_/bada_)")
    return "PASS"


def _check_scores(summary: pl.DataFrame) -> Status:
    if summary.is_empty():
        _emit("FAIL", "score_primary distribution", "summary.parquet empty")
        return "FAIL"
    n = summary.height
    null_n = summary.filter(pl.col("score_primary").is_null()).height
    nonfinite_n = summary.filter(
        ~pl.col("score_primary").is_finite() & pl.col("score_primary").is_not_null()
    ).height
    if null_n:
        _emit("FAIL", "score_primary distribution", f"{null_n}/{n} null score_primary")
        return "FAIL"
    if nonfinite_n:
        _emit("FAIL", "score_primary distribution", f"{nonfinite_n}/{n} non-finite scores")
        return "FAIL"
    stats = summary.select(
        pl.col("score_primary").min().alias("min"),
        pl.col("score_primary").max().alias("max"),
        pl.col("score_primary").mean().alias("mean"),
    ).row(0, named=True)
    msg = (
        f"{n} runs, MAE alt range=[{stats['min']:.1f}, {stats['max']:.1f}] m, "
        f"mean={stats['mean']:.1f}"
    )
    _emit("PASS", "score_primary distribution", msg)
    return "PASS"


def _print_axis_summary(summary: pl.DataFrame) -> None:
    """Mean / std / n per axis — informational, never gates the exit code."""
    if summary.is_empty():
        return
    grouped = (
        summary.group_by("axis")
        .agg(
            pl.col("score_primary").mean().alias("mean_mae"),
            pl.col("score_primary").std().alias("std_mae"),
            pl.col("score_primary").count().alias("n"),
        )
        .sort("mean_mae")
    )
    print("\n  axis effect (lower MAE = better, std across seeds):")
    with pl.Config(tbl_rows=-1, tbl_width_chars=120, float_precision=2):
        print(grouped)


def _run_dirs(results_dir: Path) -> Iterable[Path]:
    """Yield per-run output directories (skip queue.json, summary.parquet)."""
    for path in sorted(results_dir.iterdir()):
        if path.is_dir():
            yield path


def _resolve_data_dir(results_dir: Path) -> Path:
    """Read data_dir from the first run's config.yaml.

    All runs in a sweep share the same data_dir (only training hyperparams
    differ between runs), so the first one is canonical.
    """
    for run_dir in _run_dirs(results_dir):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        return Path(cfg["paths"]["data_dir"])
    msg = f"no run config.yaml found under {results_dir}"
    raise FileNotFoundError(msg)


def _aggregate_summary(results_dir: Path) -> pl.DataFrame:
    """Refresh summary.parquet by reading every metrics.json + run_config.json."""
    rows: list[dict[str, object]] = []
    for run_dir in _run_dirs(results_dir):
        m_path = run_dir / "metrics.json"
        rc_path = run_dir / "run_config.json"
        if not m_path.exists() or not rc_path.exists():
            continue
        metrics = json.loads(m_path.read_text())
        run_cfg = json.loads(rc_path.read_text())
        rows.append(
            {
                **{
                    k: run_cfg.get(k)
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
            }
        )
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    df.write_parquet(results_dir / "summary.parquet")
    return df


@app.default
def verify(
    *,
    results_dir: Annotated[
        Path,
        cyclopts.Parameter(name="--results-dir", help="Sweep output root"),
    ] = Path("results_smoke"),
) -> None:
    """Run all checks; exit 1 if any FAIL, 0 otherwise."""
    print(f"\n== sweep verification: {results_dir} ==\n")

    if not results_dir.exists():
        _emit("FAIL", "results_dir present", str(results_dir))
        sys.exit(1)

    statuses: list[Status] = []
    statuses.append(_check_queue(results_dir))
    statuses.append(_check_artifacts(results_dir))

    try:
        data_dir = _resolve_data_dir(results_dir)
    except FileNotFoundError as exc:
        _emit("FAIL", "data_dir lookup", str(exc))
        sys.exit(1)

    statuses.append(_check_isolation(results_dir, data_dir))
    statuses.append(_check_perf_parquet(results_dir, data_dir))

    summary = _aggregate_summary(results_dir)
    statuses.append(_check_scores(summary))
    _print_axis_summary(summary)

    n_fail = sum(1 for s in statuses if s == "FAIL")
    n_warn = sum(1 for s in statuses if s == "WARN")
    n_pass = sum(1 for s in statuses if s == "PASS")
    print(f"\nresult: {n_pass} pass, {n_warn} warn, {n_fail} fail\n")
    sys.exit(1 if n_fail else 0)


def main() -> None:
    """Entry point for ``python -m scripts.sweep_verify``."""
    app()


if __name__ == "__main__":
    main()
