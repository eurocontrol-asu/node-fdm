"""Visualization commands — ``fdm visualize``, ``fdm plot-performance``, ``fdm plot-example``.

Ports logic from ``scripts/opensky/08_visualize_predictions.py``,
``scripts/opensky/12_performance.py``, and ``scripts/opensky/13_example_flight.py``.

All commands require the ``[viz]`` extra: ``pip install node-fdm-pipeline[viz]``.
"""

from __future__ import annotations

from pathlib import Path

import structlog

__all__ = ["run_plot_example", "run_plot_performance", "run_visualize"]

log = structlog.get_logger()


def _require_viz() -> None:
    """Check that visualization dependencies are available.

    Raises:
        SystemExit: If matplotlib or altair are not installed.
    """
    try:
        import altair as _alt  # noqa: F401
        import matplotlib as _mpl  # noqa: F401
    except ImportError:
        msg = (
            "Visualization dependencies not found. "
            "Install with: pip install node-fdm-pipeline[viz]"
        )
        raise SystemExit(msg)  # noqa: B904


def run_visualize(
    *,
    arch: str,
    config: Path,
    typecode: str | None = None,
    flight: str | None = None,
) -> None:
    """Generate 3-panel prediction comparison plots (altitude, TAS, gamma).

    Overlays observed (red), predicted (blue), BADA (green), and selected
    (dashed black) trajectories.  Saves to ``figure_dir`` as PDF.

    Args:
        arch: Architecture identifier (``"opensky"`` or ``"qar"``).
        config: Path to YAML pipeline config.
        typecode: Single typecode to visualize (default: first from config).
        flight: Specific flight ID to visualize (default: first test flight).
    """
    _require_viz()

    import matplotlib.pyplot as plt
    import polars as pl
    from node_fdm_bada.utils import cas_to_mach, tas_to_cas
    from node_fdm_data.preprocessing.opensky import flight_processing
    from node_fdm_data.processor import FlightProcessor

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    _info = resolve_architecture(arch)

    process_dir = cfg.paths.resolve("process_dir")
    predict_dir = cfg.paths.resolve("predicted_dir")
    bada_dir = cfg.paths.resolve("bada_dir")
    figure_dir = cfg.paths.resolve("figure_dir")
    figure_dir.mkdir(parents=True, exist_ok=True)

    processor = FlightProcessor(steps=[flight_processing])
    split_df = pl.read_csv(process_dir / "dataset_split.csv")
    acft = typecode or cfg.typecodes[0]

    data_df = split_df.filter(pl.col("aircraft_type") == acft)
    test_df = data_df.filter(pl.col("split") == "test")

    log.info("visualize_start", arch=arch, typecode=acft)

    for row in test_df.iter_rows(named=True):
        flight_path = Path(row["filepath"])
        fname = flight_path.name
        flight_id = flight_path.stem

        if flight and flight_id != flight:
            continue

        bada_file = bada_dir / acft / fname
        pred_file = predict_dir / acft / fname
        if not bada_file.exists() or not pred_file.exists():
            continue

        f = pl.read_parquet(flight_path)
        processor.process(f).collect()
        f = f.hstack(pl.read_parquet(pred_file))
        f = f.hstack(pl.read_parquet(bada_file))

        # Derived CAS/Mach for BADA
        bada_cas_ms = tas_to_cas(
            f["bada_tas_ms"].to_numpy(),
            f["bada_alt_std_m"].to_numpy(),
            f["temp_k"].to_numpy(),
        )
        bada_mach = cas_to_mach(bada_cas_ms, f["bada_alt_std_m"].to_numpy())
        f = f.with_columns(
            pl.Series("bada_cas_ms", bada_cas_ms),
            pl.Series("bada_mach", bada_mach),
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.flatten()

        # Altitude
        axes[0].plot(f["alt_std_m"], color="r", label="True")
        axes[0].plot(f["pred_alt_std_m"], color="b", label="Pred")
        axes[0].plot(f["bada_alt_std_m"], color="g", label="Bada")
        if "alt_sel_m" in f.columns:
            axes[0].plot(f["alt_sel_m"], "--", color="k", lw=0.5, label="Selected")
        axes[0].set_title("Altitude [m]")

        # TAS
        axes[1].plot(f["tas_ms"], color="r", label="True")
        axes[1].plot(f["pred_tas_ms"], color="b", label="Pred")
        axes[1].plot(f["bada_tas_ms"], color="g", label="Bada")
        axes[1].set_title("True Airspeed [m/s]")

        # Gamma
        axes[2].plot(f["gamma_rad"], color="r", label="True")
        axes[2].plot(f["pred_gamma_rad"], color="b", label="Pred")
        axes[2].plot(f["bada_gamma_rad"], color="g", label="Bada")
        axes[2].set_title("Flight path angle gamma [rad]")

        for ax in axes:
            ax.grid(True, lw=0.3, linestyle="--", alpha=0.7)
            ax.legend(fontsize=8)

        plt.tight_layout()

        output_path = figure_dir / f"viz_{acft}_{flight_id}.pdf"
        fig.savefig(output_path)
        plt.close(fig)

        # Also save example parquet for use by plot-example
        f.write_parquet(cfg.paths.data_dir / "example.parquet")

        log.info("visualize_saved", path=str(output_path))

        if not flight:
            break  # Only first test flight unless --flight specified

    log.info("visualize_done", typecode=acft)


# ---------- Unit conversion constants ----------
_M_TO_FT = 3.28084
_MS_TO_KTS = 1.94384


def run_plot_performance(
    *,
    config: Path,
) -> None:
    """Generate Altair performance comparison charts per aircraft.

    Loads ``performance.parquet`` (output of ``fdm evaluate``) and generates
    horizontal bar charts grouped by flight phase with unit conversions.

    Args:
        config: Path to YAML pipeline config.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    perf_path = cfg.paths.data_dir / "performance.parquet"

    if not perf_path.exists():
        log.error("plot_performance_missing", path=str(perf_path))
        msg = f"performance.parquet not found at {perf_path}. " "Run 'fdm evaluate' first."
        raise SystemExit(msg)

    _require_viz()

    import altair as alt
    import polars as pl

    figure_dir = cfg.paths.resolve("figure_dir")
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(perf_path)

    # Rename for display
    rename_map = {
        "Altitude [m]": "altitude",
        "Flight path angle [deg]": "flight path angle",
        "True airspeed [m/s]": "true airspeed",
        "PRED": "prediction",
    }
    for old, new in rename_map.items():
        df = df.with_columns(
            pl.when(pl.col("Variable") == old)
            .then(pl.lit(new))
            .otherwise(pl.col("Variable"))
            .alias("Variable"),
            pl.when(pl.col("Model") == old)
            .then(pl.lit(new))
            .otherwise(pl.col("Model"))
            .alias("Model"),
        )

    # Unit conversions for display
    df = df.with_columns(
        pl.when(pl.col("Variable") == "altitude")
        .then(pl.col("MAE") * _M_TO_FT)
        .when(pl.col("Variable") == "true airspeed")
        .then(pl.col("MAE") * _MS_TO_KTS)
        .otherwise(pl.col("MAE"))
        .alias("MAE")
    )

    log.info("plot_performance_start", aircraft=df["Aircraft"].unique().to_list())

    base = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("MAE"),
            alt.Y("Model").title(None).axis(labelFontSize=0),
            alt.Row("Phase")
            .title(None)
            .header(
                labelOrient="top",
                labelFontSize=13,
                labelFont="Roboto Condensed",
                labelAnchor="end",
                labelAlign="right",
                labelPadding=-18,
            ),
            alt.Color("Model")
            .title(None)
            .legend(
                orient="bottom",
                labelFont="Roboto Condensed",
                labelFontSize=16,
            ),
        )
        .properties(height=20, width=200)
    )

    def _make_chart(typecode: str) -> alt.HConcatChart:
        chart = (
            alt.hconcat(
                base.transform_filter(
                    f"datum.Aircraft == '{typecode}' & datum.Variable == 'altitude'"
                ).encode(alt.X("MAE").title("altitude (in ft)")),
                base.transform_filter(
                    f"datum.Aircraft == '{typecode}' & datum.Variable == 'flight path angle'"
                ).encode(alt.X("MAE").title("flight path angle (in deg)")),
                base.transform_filter(
                    f"datum.Aircraft == '{typecode}' & datum.Variable == 'true airspeed'"
                ).encode(alt.X("MAE").title("true airspeed (in kts)")),
            )
            .properties(title=f"{typecode} performance model comparison")
            .configure_title(font="Roboto Condensed", fontSize=18, anchor="start", dy=-10)
            .configure_axisX(
                titleAnchor="start",
                titleFont="Roboto Condensed",
                titleFontSize=14,
                titleFontWeight="normal",
                labelFont="Roboto Condensed",
                labelFontSize=14,
                titlePadding=10,
            )
            .configure_facet(spacing=1)
        )
        output = figure_dir / f"performance_{typecode}.pdf"
        chart.save(output)
        log.info("plot_performance_saved", typecode=typecode, path=str(output))
        return chart

    for aircraft in df["Aircraft"].unique().to_list():
        _make_chart(aircraft)

    log.info("plot_performance_done")


def run_plot_example(
    *,
    config: Path,
) -> None:
    """Generate Altair example trajectory chart (3-panel vertical).

    Loads ``example.parquet`` (saved by ``fdm visualize``) and generates a
    three-panel chart: altitude (ft), CAS (kts), vertical speed (ft/min).

    Args:
        config: Path to YAML pipeline config.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    example_path = cfg.paths.data_dir / "example.parquet"

    if not example_path.exists():
        log.error("plot_example_missing", path=str(example_path))
        msg = f"example.parquet not found at {example_path}. " "Run 'fdm visualize' first."
        raise SystemExit(msg)

    _require_viz()

    import altair as alt
    import polars as pl

    figure_dir = cfg.paths.resolve("figure_dir")
    figure_dir.mkdir(parents=True, exist_ok=True)

    dfi = pl.read_parquet(example_path)

    log.info("plot_example_start")

    base = (
        alt.Chart(dfi)
        .mark_line()
        .encode(
            x=alt.X("timestamp").title(None).axis(titleAnchor="end", grid=False),
            color=alt.Color("renamed_source:N", title=None)
            .scale(
                domain=["selected", "predicted", "observed", "BADA"],
                range=["#79706e", "#4c78a8", "#f58518", "#54a24b"],
            )
            .legend(
                symbolStrokeWidth=8,
                orient="bottom",
                labelFont="Roboto Condensed",
                labelFontSize=16,
            ),
            strokeDash=alt.StrokeDash(
                "renamed_source:N",
                scale=alt.Scale(
                    domain=["selected", "predicted", "observed", "BADA"],
                    range=[[6, 3], [1, 0], [1, 0], [1, 0]],
                ),
                legend=None,
            ),
        )
        .properties(width=400, height=200)
    )

    chart = alt.vconcat(
        base.transform_fold(
            ["alt_std_m", "bada_alt_std_m", "pred_alt_std_m", "alt_sel_m"],
            as_=["source", "altitude"],
        )
        .transform_calculate(
            renamed_source=(
                'datum.source == "alt_std_m" ? "observed" : '
                'datum.source == "bada_alt_std_m" ? "BADA" : '
                'datum.source == "alt_sel_m" ? "selected" : "predicted"'
            )
        )
        .transform_calculate(altitude="datum.altitude / 0.3048")
        .encode(
            y=alt.Y("altitude:Q")
            .title("altitude (in ft)")
            .axis(titleAnchor="end", titleAngle=0, titleAlign="left", titleY=-10),
        ),
        base.transform_fold(
            ["cas_ms", "bada_cas_ms", "pred_cas_ms", "cas_sel_ms"],
            as_=["source", "cas"],
        )
        .transform_calculate(
            renamed_source=(
                'datum.source == "cas_ms" ? "observed" : '
                'datum.source == "bada_cas_ms" ? "BADA" : '
                'datum.source == "cas_sel_ms" ? "selected" : "predicted"'
            )
        )
        .transform_calculate(cas="datum.cas / 0.514444")
        .encode(
            y=alt.Y("cas:Q")
            .title("CAS (in kts)")
            .axis(titleAnchor="end", titleAngle=0, titleAlign="left", titleY=-10),
        ),
        base.transform_fold(
            ["vz_ms", "bada_vz_ms", "pred_vz_ms", "vz_sel_ms"],
            as_=["source", "gamma"],
        )
        .transform_calculate(
            renamed_source=(
                'datum.source == "vz_ms" ? "observed" : '
                'datum.source == "bada_vz_ms" ? "BADA" : '
                'datum.source == "vz_sel_ms" ? "selected" : "predicted"'
            )
        )
        .transform_calculate(gamma="datum.gamma * 196.850394")
        .encode(
            y=alt.Y("gamma:Q")
            .title("vertical speed (in ft/min)")
            .axis(titleAnchor="end", titleAngle=0, titleAlign="left", titleY=-10),
        ),
    ).configure_axis(
        labelFont="Roboto Condensed",
        labelFontSize=14,
        titleFont="Roboto Condensed",
        titleFontSize=18,
    )

    output = figure_dir / "traj_example.pdf"
    chart.save(output)
    log.info("plot_example_done", path=str(output))
