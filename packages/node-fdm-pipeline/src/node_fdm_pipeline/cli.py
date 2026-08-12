"""CLI entry point — cyclopts application with subcommands."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import cyclopts
import structlog

__all__ = [
    "app",
    "main",
]

log = structlog.get_logger()

app = cyclopts.App(
    name="fdm",
    help=(
        "node-fdm pipeline CLI — typed commands for flight dynamics "
        "data processing, training, and evaluation."
    ),
)


@app.command(name="version")
def version_cmd(
    *,
    _dummy: Annotated[str | None, cyclopts.Parameter(show=False)] = None,
) -> None:
    """Print the package version and exit."""
    try:
        v = version("node-fdm-pipeline")
    except PackageNotFoundError:
        v = "0.0.0+dev"
    print(f"node-fdm-pipeline {v}")  # noqa: T201


@app.command
def train(
    *,
    arch: Annotated[
        str,
        cyclopts.Parameter(help="Installed architecture alias or canonical name"),
    ],
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    typecode: Annotated[
        str | None,
        cyclopts.Parameter(help="Single ICAO typecode to train (default: all from config)"),
    ] = None,
    epochs: Annotated[
        int | None,
        cyclopts.Parameter(help="Override number of training epochs"),
    ] = None,
    batch_size: Annotated[
        int | None,
        cyclopts.Parameter(name="--batch-size", help="Override batch size"),
    ] = None,
    lr: Annotated[
        float | None,
        cyclopts.Parameter(help="Override learning rate"),
    ] = None,
    method: Annotated[
        str,
        cyclopts.Parameter(help="ODE integration method: euler or rk4"),
    ] = "euler",
    seq_len: Annotated[
        int | None,
        cyclopts.Parameter(help="Override sequence length for training windows"),
    ] = None,
    shift: Annotated[
        int | None,
        cyclopts.Parameter(help="Override shift between windows (defaults to seq-len)"),
    ] = None,
    device: Annotated[
        str,
        cyclopts.Parameter(help="PyTorch device for training"),
    ] = "cpu",
    model_name: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--model-name",
            help="Custom model name (default: {arch}_{typecode})",
        ),
    ] = None,
    lambda_tracking: Annotated[
        float | None,
        cyclopts.Parameter(
            name="--lambda-tracking",
            help="Tracking loss weight on autopilot targets (0=disabled)",
        ),
    ] = None,
    use_mode_weights: Annotated[
        bool | None,
        cyclopts.Parameter(
            name="--use-mode-weights",
            negative="--no-use-mode-weights",
            negative_none=(),
            help=(
                "Override cfg.training.use_mode_weights. When unset (default), "
                "the YAML value is used; CLI flag wins over YAML."
            ),
        ),
    ] = None,
    train_limit: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--train-limit",
            help="Max training samples (default: 5000)",
        ),
    ] = None,
    activation: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--activation",
            help="Hidden-layer activation: silu (default), relu, gelu, or tanh",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--seed",
            help="Seed torch/numpy/random for reproducible runs",
        ),
    ] = None,
    mode_weight_alpha: Annotated[
        float | None,
        cyclopts.Parameter(
            name="--mode-weight-alpha",
            help="Override cfg.training.mode_weight_alpha (power-law exponent in [0,1])",
        ),
    ] = None,
    backbone_depth: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--backbone-depth",
            help="Number of hidden layers in the shared backbone MLP (default: 3)",
        ),
    ] = None,
    head_depth: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--head-depth",
            help="Number of hidden layers per output head (default: 2)",
        ),
    ] = None,
    hidden_width: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--hidden-width",
            help="Hidden-layer width (neurons) for backbone and heads (default: 48)",
        ),
    ] = None,
) -> None:
    """Train Neural ODE models for aircraft flight dynamics."""
    from node_fdm_pipeline.commands.train import run_training

    run_training(
        arch=arch,
        config=config,
        typecode=typecode,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        method=method,
        seq_len=seq_len,
        shift=shift,
        device=device,
        model_name=model_name,
        lambda_tracking=lambda_tracking,
        use_mode_weights=use_mode_weights,
        train_limit=train_limit,
        activation=activation,
        seed=seed,
        mode_weight_alpha=mode_weight_alpha,
        backbone_depth=backbone_depth,
        head_depth=head_depth,
        hidden_width=hidden_width,
    )


@app.command
def resume(
    *,
    model: Annotated[
        Path,
        cyclopts.Parameter(help="Path to model directory containing meta.json"),
    ],
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    epochs: Annotated[
        int | None,
        cyclopts.Parameter(help="Override number of training epochs"),
    ] = None,
    batch_size: Annotated[
        int | None,
        cyclopts.Parameter(name="--batch-size", help="Override batch size"),
    ] = None,
    lr: Annotated[
        float | None,
        cyclopts.Parameter(help="Override learning rate"),
    ] = None,
    seq_len: Annotated[
        int | None,
        cyclopts.Parameter(help="Override sequence length for training windows"),
    ] = None,
    shift: Annotated[
        int | None,
        cyclopts.Parameter(help="Override shift between windows"),
    ] = None,
    overwrite: Annotated[
        bool,
        cyclopts.Parameter(help="Save back into the same model directory"),
    ] = False,
    device: Annotated[
        str,
        cyclopts.Parameter(help="PyTorch device for training"),
    ] = "cpu",
    method: Annotated[
        str | None,
        cyclopts.Parameter(help="Override ODE integration method: euler or rk4"),
    ] = None,
    model_name: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--model-name",
            help="Custom output model name (default: same as source model)",
        ),
    ] = None,
    lambda_tracking: Annotated[
        float | None,
        cyclopts.Parameter(
            name="--lambda-tracking",
            help="Tracking loss weight on autopilot targets (0=disabled)",
        ),
    ] = None,
    reset_loss: Annotated[
        bool,
        cyclopts.Parameter(
            name="--reset-loss",
            help="Ignore saved best_val_loss and start fresh from inf",
        ),
    ] = False,
    typecode: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Aircraft typecode (overrides parsing from model dir name)",
        ),
    ] = None,
    use_mode_weights: Annotated[
        bool | None,
        cyclopts.Parameter(
            name="--use-mode-weights",
            negative="--no-use-mode-weights",
            negative_none=(),
            help=(
                "Override cfg.training.use_mode_weights. When unset (default), "
                "the YAML value is used; CLI flag wins over YAML."
            ),
        ),
    ] = None,
) -> None:
    """Resume training from an existing model checkpoint."""
    from node_fdm_pipeline.commands.resume import run_resume

    run_resume(
        model=model,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seq_len=seq_len,
        shift=shift,
        overwrite=overwrite,
        device=device,
        method=method,
        model_name=model_name,
        lambda_tracking=lambda_tracking,
        reset_loss=reset_loss,
        typecode=typecode,
        use_mode_weights=use_mode_weights,
    )


@app.command
def predict(
    *,
    arch: Annotated[
        str,
        cyclopts.Parameter(help="Installed architecture alias or canonical name"),
    ],
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    typecode: Annotated[
        str | None,
        cyclopts.Parameter(help="Single ICAO typecode to predict"),
    ] = None,
    device: Annotated[
        str,
        cyclopts.Parameter(help="PyTorch device for inference"),
    ] = "cpu",
    local_model: Annotated[
        bool,
        cyclopts.Parameter(name="--local-model", help="Use local model directory"),
    ] = False,
    model_name: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--model-name",
            help=(
                "Local checkpoint directory name relative to models_dir "
                "(default: {arch_registry_name}_{typecode}). Only used with --local-model."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--limit",
            help="Predict at most N flights per typecode (default: all).",
        ),
    ] = None,
) -> None:
    """Predict flight trajectories using trained Neural ODE models."""
    from node_fdm_pipeline.commands.predict import run_predict

    run_predict(
        arch=arch,
        config=config,
        typecode=typecode,
        device=device,
        local_model=local_model,
        model_name=model_name,
        limit=limit,
    )


@app.command(name="predict-bada")
def predict_bada(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    typecode: Annotated[
        str | None,
        cyclopts.Parameter(help="Single ICAO typecode to predict"),
    ] = None,
    jobs: Annotated[
        int | None,
        cyclopts.Parameter(help="Number of parallel workers"),
    ] = None,
) -> None:
    """Run BADA 4.2 baseline predictions."""
    from node_fdm_pipeline.commands.predict import run_predict_bada

    run_predict_bada(
        config=config,
        typecode=typecode,
        jobs=jobs,
    )


@app.command
def evaluate(
    *,
    arch: Annotated[
        str,
        cyclopts.Parameter(help="Architecture: opensky or qar"),
    ],
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    model_name: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--model-name",
            help=(
                "Checkpoint directory name written by 'fdm train --model-name'. "
                "When set, predictions are read from "
                "<predict_dir>/<model_name>/<typecode>/ and metrics written to "
                "<data_dir>/model_performance/<model_name>/. Defaults to <arch>."
            ),
        ),
    ] = None,
) -> None:
    """Compute prediction error metrics per flight phase."""
    from node_fdm_pipeline.commands.evaluate import run_evaluate

    run_evaluate(arch=arch, config=config, model_name=model_name)


@app.command(name="aircraft-list")
def aircraft_list_cmd(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    sample_size: Annotated[
        int,
        cyclopts.Parameter(name="--sample-size", help="Max flights per typecode"),
    ] = 100,
    query_date: Annotated[
        str,
        cyclopts.Parameter(name="--query-date", help="Start date to query (YYYY-MM-DD)"),
    ] = "2025-10-01",
    query_end_date: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--query-end-date",
            help="End date (exclusive, YYYY-MM-DD); defaults to query_date + 1 day",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Query OpenSky for aircraft database and save CSV."""
    from node_fdm_pipeline.commands.data import aircraft_list

    aircraft_list(
        config=config,
        sample_size=sample_size,
        query_date=query_date,
        query_end_date=query_end_date,
        dry_run=dry_run,
    )


@app.command
def download(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    start_date: Annotated[
        str,
        cyclopts.Parameter(name="--start-date", help="Start date (YYYY-MM-DD)"),
    ] = "",
    end_date: Annotated[
        str,
        cyclopts.Parameter(name="--end-date", help="End date (YYYY-MM-DD)"),
    ] = "",
    flight_plan: Annotated[
        Path | None,
        cyclopts.Parameter(
            name="--flight-plan",
            help="CSV of selected flights (icao24 + day); fetch only those aircraft-days",
        ),
    ] = None,
    step_hours: Annotated[
        int,
        cyclopts.Parameter(name="--step-hours", help="Hours between windows"),
    ] = 24,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
    no_decode: Annotated[
        bool,
        cyclopts.Parameter(name="--no-decode", help="Populate raw cache only; skip decode"),
    ] = False,
    force_refresh: Annotated[
        bool,
        cyclopts.Parameter(name="--force-refresh", help="Bypass cache and re-fetch all data"),
    ] = False,
) -> None:
    """Download ADS-B history from OpenSky, by date range or by flight plan.

    Range mode fetches every aircraft in aircraft_db.csv on every day of the
    range. Plan mode fetches, per day, only the aircraft that have a selected
    flight that day — so the cost follows the selection rather than the span.
    """
    from node_fdm_pipeline.commands.data import download as download_fn

    download_fn(
        config=config,
        start_date=start_date,
        end_date=end_date,
        step_hours=step_hours,
        dry_run=dry_run,
        no_decode=no_decode,
        force_refresh=force_refresh,
        flight_plan=flight_plan,
    )


@app.command(name="download-fleet")
def download_fleet(
    *,
    fleet_dir: Annotated[
        Path,
        cyclopts.Parameter(
            name="--fleet-dir", help="Fleet directory holding types/*/config*.yaml"
        ),
    ],
    workers: Annotated[
        int,
        cyclopts.Parameter(
            name="--workers",
            help="Concurrent Trino queries; the cluster caps this per account",
        ),
    ] = 2,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate the plan without I/O"),
    ] = False,
    force_refresh: Annotated[
        bool,
        cyclopts.Parameter(name="--force-refresh", help="Bypass cache and re-fetch all data"),
    ] = False,
) -> None:
    """Download every cohort at once: aircraft mutualised per date, dates in parallel.

    ``download --flight-plan`` runs one cohort at a time and asks for that
    cohort's 2-3 aircraft per day, so the same calendar day is fetched once per
    cohort. This visits each date once instead, naming every aircraft any cohort
    wants that day, and routes the rows back into the owning silo — 2,545
    requests where the per-cohort loop issues 35,000, for the identical payload.

    Dates are independent, so several run concurrently. Keep ``--workers`` at or
    below the account's Trino concurrency limit; above it, queries are rejected
    rather than queued. Decoding is not chained here — run ``decode`` per cohort
    afterwards, which needs no network.
    """
    from node_fdm_pipeline.commands._fleet_fetch import download_fleet as run_fleet
    from node_fdm_pipeline.commands._fleet_plan import build_fleet_plan, discover_cohorts

    plan = build_fleet_plan(discover_cohorts(fleet_dir))
    run_fleet(plan, workers=workers, force=force_refresh, dry_run=dry_run)


@app.command(name="split-from-selection")
def split_from_selection(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to the cohort's YAML config file"),
    ],
    selection: Annotated[
        Path,
        cyclopts.Parameter(name="--selection", help="Path to the cohort's selection_*.csv"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Report coverage without writing"),
    ] = False,
) -> None:
    """Write ``meta_split`` from the stratified selection — replaces ``split``.

    ``split`` assigns train/val/test from a hash of ``raw_icao24``. In the v2
    pipeline the split is already drawn, before anything is fetched, by
    ``stratify.py`` — keyed on **MSN**, and stratified across year, hour,
    duration and region. Re-deriving it here would discard that, and hashing
    icao24 would split an airframe across sets: 103 MSNs in this fleet carry more
    than one Mode-S address.

    This carries the decided split onto the table instead. It refuses rather than
    guesses: an icao24 under two splits, or a row left without one, is an error.
    """
    from node_fdm_pipeline.commands._split_from_selection import (
        split_from_selection as run_split,
    )

    run_split(config=config, selection=selection, dry_run=dry_run)


@app.command(name="decode-fleet")
def decode_fleet(
    *,
    fleet_dir: Annotated[
        Path,
        cyclopts.Parameter(
            name="--fleet-dir", help="Fleet directory holding types/*/config*.yaml"
        ),
    ],
    workers: Annotated[
        int,
        cyclopts.Parameter(
            name="--workers", help="Concurrent decode processes; each peaks near 95-160 GiB"
        ),
    ] = 1,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="List the spans without decoding"),
    ] = False,
) -> None:
    """Decode every cohort's raw cache into its Delta table, several at a time.

    ``download-fleet`` does not chain this: being date-major, it completes no
    cohort before the final date, so there is nothing to decode incrementally.
    Decoding is a separate phase and a different bottleneck — no network at all,
    so it scales with local cores rather than with the Trino quota.

    Each cohort decodes its own span, derived from its own selection. The limit is
    memory, not cores: one decode of the smallest cohort peaks near 94 GiB and the
    largest is 1.68x that, so ``--workers`` is a RAM budget — it defaults to 1 and
    the command warns when the estimate exceeds what is free.
    """
    from node_fdm_pipeline.commands._fleet_decode import decode_fleet as run_decode_fleet

    run_decode_fleet(fleet_dir, workers=workers, dry_run=dry_run)


@app.command
def decode(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    start_date: Annotated[
        str,
        cyclopts.Parameter(name="--start-date", help="Start date (YYYY-MM-DD)"),
    ],
    end_date: Annotated[
        str,
        cyclopts.Parameter(name="--end-date", help="End date (YYYY-MM-DD)"),
    ],
    icao24_filter: Annotated[
        Path | None,
        cyclopts.Parameter(
            name="--icao24-filter",
            help="Optional file with one icao24 per line; intersected with aircraft_db.csv",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Rebuild the Delta Table from the raw parquet cache (no network)."""
    from node_fdm_pipeline.commands.data import decode as decode_fn

    decode_fn(
        config=config,
        start_date=start_date,
        end_date=end_date,
        icao24_filter=icao24_filter,
        dry_run=dry_run,
    )


@app.command
def enrich(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Enrich the Delta Table with ERA5 weather data (étape 3)."""
    from node_fdm_pipeline.commands.data import enrich as enrich_fn

    enrich_fn(config=config, dry_run=dry_run)


@app.command
def derive(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Compute derived physics columns (fdm_gamma_rad, fdm_long_wind_kt, etc.) — étape 4."""
    from node_fdm_pipeline.commands.data import derive as derive_fn

    derive_fn(config=config, dry_run=dry_run)


@app.command(name="label-modes")
def label_modes(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Attach the per-sample mode label (TURN + 12 vert x long classes) — étape 5."""
    from node_fdm_pipeline.commands.data import label_modes as label_modes_fn

    label_modes_fn(config=config, dry_run=dry_run)


@app.command(name="clean-speeds")
def clean_speeds(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Clean BDS speed signals via Hampel + ERA fill — étape 4 (pre-derive)."""
    from node_fdm_pipeline.commands.data import clean_speeds as clean_speeds_fn

    clean_speeds_fn(config=config, dry_run=dry_run)


@app.command
def segments(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Detect constant segments and build fdm_*_sel columns — étape 5."""
    from node_fdm_pipeline.commands.data import segments as segments_fn

    segments_fn(config=config, dry_run=dry_run)


@app.command
def convert(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Convert to SI units and compute temporal derivatives — étapes 6-7."""
    from node_fdm_pipeline.commands.data import convert as convert_fn

    convert_fn(config=config, dry_run=dry_run)


@app.command
def preprocess(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Resample to regular grid with gap-aware interpolation (étape 1.5)."""
    from node_fdm_pipeline.commands.data import preprocess as preprocess_fn

    preprocess_fn(config=config, dry_run=dry_run)


@app.command
def flag(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Add validity flag columns (fdm_flag_*) to the Delta Table."""
    from node_fdm_pipeline.commands.data import flag as flag_fn

    flag_fn(config=config, dry_run=dry_run)


@app.command
def split(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    train_ratio: Annotated[
        float,
        cyclopts.Parameter(name="--train-ratio", help="Train split ratio"),
    ] = 0.7,
    val_ratio: Annotated[
        float,
        cyclopts.Parameter(name="--val-ratio", help="Validation split ratio"),
    ] = 0.15,
    test_ratio: Annotated[
        float,
        cyclopts.Parameter(name="--test-ratio", help="Test split ratio"),
    ] = 0.15,
    seed: Annotated[
        int,
        cyclopts.Parameter(help="Hash salt for reproducible splits"),
    ] = 42,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Assign train/val/test split to each row by icao24 hash (étape 8)."""
    from node_fdm_pipeline.commands.data import split as split_fn

    split_fn(
        config=config,
        ratios=(train_ratio, val_ratio, test_ratio),
        seed=seed,
        dry_run=dry_run,
    )


@app.command
def identify(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    gap_threshold_s: Annotated[
        int,
        cyclopts.Parameter(
            name="--gap-threshold",
            help="Gap threshold in seconds for segment splitting",
        ),
    ] = 30,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Identify flights: segment at gaps, assign IDs, join flightlist metadata."""
    from node_fdm_pipeline.commands.data import identify as identify_fn

    identify_fn(
        config=config,
        gap_threshold_s=gap_threshold_s,
        dry_run=dry_run,
    )


@app.command(name="dataset-stats")
def dataset_stats(
    *,
    arch: Annotated[
        str,
        cyclopts.Parameter(help="Architecture: opensky or qar"),
    ],
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
) -> None:
    """Compute dataset statistics (flight counts, hours per split)."""
    from node_fdm_pipeline.commands.stats import run_dataset_stats

    run_dataset_stats(arch=arch, config=config)


@app.command
def visualize(
    *,
    arch: Annotated[
        str,
        cyclopts.Parameter(help="Architecture: opensky or qar"),
    ],
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    typecode: Annotated[
        str | None,
        cyclopts.Parameter(help="ICAO typecode"),
    ] = None,
    flight: Annotated[
        str | None,
        cyclopts.Parameter(help="Specific flight ID to visualize"),
    ] = None,
    limit: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--limit",
            help="Visualize at most N flights per typecode (default: all).",
        ),
    ] = None,
) -> None:
    """Visualize Node-FDM inference vs ground truth (4x2 figure per flight)."""
    from node_fdm_pipeline.commands.visualize import run_visualize

    run_visualize(arch=arch, config=config, typecode=typecode, flight=flight, limit=limit)


@app.command(name="plot-performance")
def plot_performance(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
) -> None:
    """Generate Altair performance comparison charts per aircraft."""
    from node_fdm_pipeline.commands.visualize import run_plot_performance

    run_plot_performance(config=config)


@app.command(name="table-info")
def table_info_cmd(
    *,
    table_path: Annotated[
        Path,
        cyclopts.Parameter(name="--table-path", help="Path to the Delta table directory"),
    ] = Path("data/flights.delta"),
) -> None:
    """Inspect a Delta table: partitions, columns, and version count."""
    from node_fdm_data.delta import table_info

    info = table_info(table_path)
    print(f"Partitions ({len(info['partitions'])}):")  # noqa: T201
    for p in info["partitions"]:
        print(f"  - {p}")  # noqa: T201
    print(f"\nColumns ({len(info['columns'])}):")  # noqa: T201
    for c in info["columns"]:
        print(f"  - {c}")  # noqa: T201
    print(f"\nVersions: {info['versions']}")  # noqa: T201


@app.command(name="plot-example")
def plot_example(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
) -> None:
    """Generate Altair example trajectory chart."""
    from node_fdm_pipeline.commands.visualize import run_plot_example

    run_plot_example(config=config)


def main() -> None:
    """Entry point for the ``fdm`` CLI."""
    app()
