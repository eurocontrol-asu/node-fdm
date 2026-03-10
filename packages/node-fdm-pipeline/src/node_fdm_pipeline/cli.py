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
    device: Annotated[
        str,
        cyclopts.Parameter(help="PyTorch device for training"),
    ] = "cpu",
) -> None:
    """Train Neural ODE models for aircraft flight dynamics."""
    import structlog

    from node_fdm_pipeline.config import PipelineConfig

    log = structlog.get_logger()
    cfg = PipelineConfig.from_yaml(config)
    typecodes = [typecode] if typecode else cfg.typecodes
    log.info(
        "train_start",
        arch=arch,
        typecodes=typecodes,
        device=device,
        config=str(config),
    )
    log.info("train_placeholder", msg="Training commands will be implemented in AXM-363")


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
    import structlog

    from node_fdm_pipeline.config import PipelineConfig

    log = structlog.get_logger()
    cfg = PipelineConfig.from_yaml(config)
    typecodes = [typecode] if typecode else cfg.typecodes
    log.info(
        "predict_start",
        arch=arch,
        typecodes=typecodes,
        device=device,
        local_model=local_model,
    )
    log.info("predict_placeholder", msg="Predict commands will be implemented in AXM-363")


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
    import structlog

    from node_fdm_pipeline.config import PipelineConfig

    log = structlog.get_logger()
    cfg = PipelineConfig.from_yaml(config)
    typecodes = [typecode] if typecode else cfg.typecodes
    log.info("predict_bada_start", typecodes=typecodes, jobs=jobs)
    log.info("predict_bada_placeholder", msg="BADA commands will be implemented in AXM-363")


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
    import structlog

    from node_fdm_pipeline.config import PipelineConfig

    log = structlog.get_logger()
    _cfg = PipelineConfig.from_yaml(config)
    log.info("evaluate_start", arch=arch)
    log.info("evaluate_placeholder", msg="Evaluate commands will be implemented in AXM-363")


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
def preprocess(
    *,
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    history_file: Annotated[
        Path,
        cyclopts.Parameter(name="--history-file", help="Path to history_*.parquet"),
    ],
    workers: Annotated[
        int,
        cyclopts.Parameter(help="Number of parallel workers"),
    ] = 1,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Preprocess a raw ADS-B history file (EHS decode, filter, resample)."""
    from node_fdm_pipeline.commands.data import preprocess as preprocess_fn

    preprocess_fn(
        config=config,
        history_file=history_file,
        workers=workers,
        dry_run=dry_run,
    )


@app.command
def process(
    *,
    arch: Annotated[
        str,
        cyclopts.Parameter(help="Architecture: opensky or qar"),
    ],
    config: Annotated[
        Path,
        cyclopts.Parameter(help="Path to YAML config file"),
    ],
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(name="--dry-run", help="Validate without I/O"),
    ] = False,
) -> None:
    """Process preprocessed flight data and create train/val/test split."""
    from node_fdm_pipeline.commands.data import process as process_fn

    process_fn(arch=arch, config=config, dry_run=dry_run)


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
    import structlog

    from node_fdm_pipeline.config import PipelineConfig

    log = structlog.get_logger()
    _cfg = PipelineConfig.from_yaml(config)
    log.info("dataset_stats_start", arch=arch)
    log.info("dataset_stats_placeholder", msg="Stats commands will be implemented in AXM-364")


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
    import structlog

    from node_fdm_pipeline.config import PipelineConfig

    log = structlog.get_logger()
    _cfg = PipelineConfig.from_yaml(config)
    log.info("visualize_start", arch=arch, typecode=typecode, flight=flight)
    log.info("visualize_placeholder", msg="Visualization commands will be implemented in AXM-364")


def main() -> None:
    """Entry point for the ``fdm`` CLI."""
    app()
