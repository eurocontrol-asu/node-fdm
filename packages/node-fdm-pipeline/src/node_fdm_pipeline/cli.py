"""CLI entry point — cyclopts application with subcommands."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import cyclopts

__all__ = [
    "app",
    "main",
]

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
        cyclopts.Parameter(help="Architecture: opensky or qar"),
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
    device: Annotated[
        str,
        cyclopts.Parameter(help="PyTorch device for training"),
    ] = "cpu",
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
        device=device,
    )


@app.command
def predict(
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
) -> None:
    """Predict flight trajectories using trained Neural ODE models."""
    from node_fdm_pipeline.commands.predict import run_predict

    run_predict(
        arch=arch,
        config=config,
        typecode=typecode,
        device=device,
        local_model=local_model,
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
) -> None:
    """Compute prediction error metrics per flight phase."""
    from node_fdm_pipeline.commands.evaluate import run_evaluate

    run_evaluate(arch=arch, config=config)


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
        cyclopts.Parameter(name="--query-date", help="Date to query (YYYY-MM-DD)"),
    ] = "2025-10-01",
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
    ],
    end_date: Annotated[
        str,
        cyclopts.Parameter(name="--end-date", help="End date (YYYY-MM-DD)"),
    ],
    step_hours: Annotated[
        int,
        cyclopts.Parameter(name="--step-hours", help="Hours between windows"),
    ] = 24,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Download ADS-B history data from OpenSky by date range."""
    from node_fdm_pipeline.commands.data import download as download_fn

    download_fn(
        config=config,
        start_date=start_date,
        end_date=end_date,
        step_hours=step_hours,
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
) -> None:
    """Visualize prediction comparisons (Node-FDM vs BADA vs ground truth)."""
    from node_fdm_pipeline.commands.visualize import run_visualize

    run_visualize(arch=arch, config=config, typecode=typecode, flight=flight)


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
    from node_fdm_pipeline.commands.table_info import table_info

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
