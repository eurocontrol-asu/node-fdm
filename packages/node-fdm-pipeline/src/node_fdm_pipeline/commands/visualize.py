"""Visualization commands — ``fdm visualize``, ``fdm plot-performance``, ``fdm plot-example``.

Ports logic from ``scripts/opensky/08_visualize_predictions.py``,
``scripts/opensky/12_performance.py``, and ``scripts/opensky/13_example_flight.py``.

All commands require the ``[viz]`` extra: ``pip install node-fdm-pipeline[viz]``.
"""
# ruff: noqa: RUF001, RUF002, RUF003
# Greek letters (γ, ψ) and the × multiplication sign appear in physics-flavoured
# axis labels and docstrings — they are intentional, not OCR artefacts.

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

__all__ = ["run_plot_example", "run_plot_performance", "run_visualize"]

log = structlog.get_logger()


def _require_viz() -> None:
    """Check that visualization dependencies are available.

    Raises:
        SystemExit: If matplotlib or altair are not installed.
    """
    try:
        import altair as _alt  # noqa: F401  # type: ignore[import-not-found,unused-ignore]
        import matplotlib as _mpl  # noqa: F401
    except ImportError:
        msg = (
            "Visualization dependencies not found. "
            "Install with: pip install node-fdm-pipeline[viz]"
        )
        raise SystemExit(msg)  # noqa: B904


_STEP_S: float = 4.0
"""Time-step (s) between samples on the Delta Table grid (v3 pipeline)."""

_MIN_FINITE_ROWS: int = 2
"""Minimum number of finite-mask rows required to render a flight."""


def _set_ylim(ax: Any, true_vals: Any) -> None:
    """Set y-axis limits with a 10% margin around finite values of ``true_vals``."""
    import numpy as np

    valid = true_vals[np.isfinite(true_vals)]
    if len(valid) == 0:
        return
    ymin, ymax = valid.min(), valid.max()
    margin = (ymax - ymin) * 0.10 if ymax != ymin else abs(ymax) * 0.10 + 1.0
    ax.set_ylim(ymin - margin, ymax + margin)


def _shade_unknown(ax: Any, t: Any, mask: Any, label: str) -> None:
    """Shade rows where target is absent (unknown / no detected segment)."""
    if not mask.any():
        return
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t, ymin, ymax, where=mask, alpha=0.08, color="gray", label=label)
    ax.set_ylim(ymin, ymax)


def _integrate_ground_track(
    *,
    lat0: float,
    lon0: float,
    heading_pred: Any,
    tas_pred: Any,
    gamma_pred: Any,
    u_wind: Any,
    v_wind: Any,
    step_s: float,
) -> tuple[Any, Any]:
    """Integrate the predicted ground-track from heading + TAS + γ + wind.

    Wind triangle: ``ground_velocity = TAS_horiz * (sin ψ, cos ψ) + (u_wind, v_wind)``.
    """
    import numpy as np

    r_earth_m = 6_371_000.0
    n_p = len(heading_pred)
    tas_horiz = tas_pred * np.cos(gamma_pred)
    v_e = tas_horiz * np.sin(heading_pred) + u_wind[:n_p]
    v_n = tas_horiz * np.cos(heading_pred) + v_wind[:n_p]
    lat_pred = np.empty(n_p, dtype=np.float64)
    lon_pred = np.empty(n_p, dtype=np.float64)
    lat_pred[0] = lat0
    lon_pred[0] = lon0
    for k in range(1, n_p):
        lat_rad_k = np.radians(lat_pred[k - 1])
        d_lat_deg = np.degrees(v_n[k - 1] * step_s / r_earth_m)
        d_lon_deg = np.degrees(v_e[k - 1] * step_s / (r_earth_m * max(np.cos(lat_rad_k), 1e-6)))
        lat_pred[k] = lat_pred[k - 1] + d_lat_deg
        lon_pred[k] = lon_pred[k - 1] + d_lon_deg
    return lat_pred, lon_pred


def _plot_inference_figure(  # noqa: PLR0912
    *,
    flight_id: str,
    info: object,
    flight_df: object,
    pred_df: object,
    output_path: Path,
) -> None:
    """Render the 4×2 inference figure for one flight and save it as PNG.

    Layout:
        Row 0: Heading    | Altitude
        Row 1: TAS        | FPA (γ)
        Row 2: GroundTrack| VZ
        Row 3: CAS        | Mach
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from node_fdm_data.physics.constants import GAMMA_AIR, R
    from node_fdm_data.physics.speed import tas_to_cas_real

    has_lateral = "fdm_heading_rad" in info.x_cols  # type: ignore[attr-defined]
    kt_to_ms = 0.514444

    # ── Build numpy arrays from the Delta + predict join ───────────────
    x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)  # type: ignore[attr-defined]
    u_arr = flight_df.select(info.u_cols).to_numpy().astype(np.float32)  # type: ignore[attr-defined]
    e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)  # type: ignore[attr-defined]

    gamma_known_idx = info.u_cols.index("fdm_gamma_target_known")  # type: ignore[attr-defined]
    gamma_known = u_arr[:, gamma_known_idx].copy()

    # Match `_filter_nan_segments` (predict.py): filter on x AND e only.
    # u_cols are not constrained — sel/target columns can be NaN outside
    # detected segments and the matching `_known` masks carry the info.
    finite_mask = np.isfinite(x_arr).all(axis=1) & np.isfinite(e_arr).all(axis=1)
    if finite_mask.sum() < _MIN_FINITE_ROWS:
        log.warning("visualize_skip_no_finite_rows", flight_id=flight_id)
        return

    x_arr = x_arr[finite_mask]
    u_arr = u_arr[finite_mask]
    e_arr = e_arr[finite_mask]
    gamma_known = gamma_known[finite_mask]

    # Pred arrays — already pre-filtered when written; if their length matches
    # the Delta's filtered length we use them as-is, otherwise we skip the flight.
    pred_alt_full = pred_df["pred_raw_alt_m"].to_numpy()  # type: ignore[index]
    if len(pred_alt_full) != len(x_arr):
        log.warning(
            "visualize_skip_length_mismatch",
            flight_id=flight_id,
            delta_rows=len(x_arr),
            pred_rows=len(pred_alt_full),
        )
        return
    pred_alt = pred_alt_full
    pred_tas = pred_df["pred_era_tas_ms"].to_numpy()  # type: ignore[index]
    pred_gamma = pred_df["pred_fdm_gamma_rad"].to_numpy()  # type: ignore[index]
    pred_heading = (
        pred_df["pred_fdm_heading_rad"].to_numpy()  # type: ignore[index]
        if has_lateral and "pred_fdm_heading_rad" in pred_df.columns  # type: ignore[attr-defined]
        else None
    )

    n = x_arr.shape[0]
    time_true = np.arange(n) * _STEP_S / 60  # minutes
    time_pred = time_true  # same grid (predict parquet already aligned)

    alt_idx = info.x_cols.index("raw_alt_m")  # type: ignore[attr-defined]
    tas_idx = info.x_cols.index("era_tas_ms")  # type: ignore[attr-defined]
    gamma_idx = info.x_cols.index("fdm_gamma_rad")  # type: ignore[attr-defined]
    alt_true = x_arr[:, alt_idx]
    tas_true = x_arr[:, tas_idx]
    gamma_true = x_arr[:, gamma_idx]

    heading_true = None
    heading_target = None
    heading_target_known = None
    if has_lateral:
        h_idx = info.x_cols.index("fdm_heading_rad")  # type: ignore[attr-defined]
        heading_true = x_arr[:, h_idx]
        h_target_idx = info.u_cols.index("fdm_heading_target_rad")  # type: ignore[attr-defined]
        h_known_idx = info.u_cols.index("fdm_heading_target_known")  # type: ignore[attr-defined]
        heading_target = u_arr[:, h_target_idx]
        heading_target_known = u_arr[:, h_known_idx]

    alt_target = u_arr[:, info.u_cols.index("fdm_alt_target_m")]  # type: ignore[attr-defined]
    tas_target = u_arr[:, info.u_cols.index("fdm_tas_target_ms")]  # type: ignore[attr-defined]
    gamma_target_raw = u_arr[:, info.u_cols.index("fdm_gamma_target_rad")]  # type: ignore[attr-defined]
    gamma_target = np.where(gamma_known == 1.0, gamma_target_raw, np.nan)

    # ── Extras (true Mach/CAS/VZ + sel + lat/lon/in_turn for lateral) ──
    extra_cols = [
        "era_mach",
        "fdm_mach_sel",
        "bds_ias_ms",
        "fdm_cas_sel_kt",
        "raw_vz_ms",
        "fdm_vz_sel_ms",
        "fdm_tas_target_known",
        "era_temp_K",
    ]
    if has_lateral:
        extra_cols += ["raw_lat_deg", "raw_lon_deg", "fdm_in_turn"]
    extra = flight_df.select(extra_cols).to_numpy().astype(np.float32)[finite_mask]  # type: ignore[attr-defined]
    mach_true = extra[:, 0]
    mach_sel = extra[:, 1]
    cas_true = extra[:, 2]
    cas_sel = extra[:, 3] * kt_to_ms
    vz_true = extra[:, 4]
    vz_sel = extra[:, 5]
    tas_known_mask = extra[:, 6]
    temp_true = extra[:, 7]
    lat_arr = extra[:, 8] if has_lateral else None
    lon_arr = extra[:, 9] if has_lateral else None
    in_turn_arr = extra[:, 10].astype(bool) if has_lateral else None

    mach_unknown = np.isnan(mach_sel)
    cas_unknown = np.isnan(cas_sel)
    vz_unknown = np.isnan(vz_sel)
    tas_unknown = tas_known_mask == 0.0

    mach_pred = pred_tas / np.sqrt(GAMMA_AIR * R * temp_true)
    cas_pred = tas_to_cas_real(pred_tas, pred_alt, temp_true)
    vz_pred = pred_tas * np.sin(pred_gamma)

    lat_pred: Any = None
    lon_pred: Any = None
    if has_lateral and pred_heading is not None and lat_arr is not None and lon_arr is not None:
        u_wind = e_arr[:, info.e0_cols.index("era_u_wind_ms")]  # type: ignore[attr-defined]
        v_wind = e_arr[:, info.e0_cols.index("era_v_wind_ms")]  # type: ignore[attr-defined]
        lat_pred, lon_pred = _integrate_ground_track(
            lat0=float(lat_arr[0]),
            lon0=float(lon_arr[0]),
            heading_pred=pred_heading,
            tas_pred=pred_tas,
            gamma_pred=pred_gamma,
            u_wind=u_wind,
            v_wind=v_wind,
            step_s=_STEP_S,
        )

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        cartopy_available = True
    except ImportError:
        cartopy_available = False

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for r_src in (1, 3):
        axes[r_src, 0].sharex(axes[0, 0])
    for r_src in (1, 2, 3):
        axes[r_src, 1].sharex(axes[0, 1])

    # (0,0) Heading
    ax = axes[0, 0]
    if (
        has_lateral
        and heading_true is not None
        and pred_heading is not None
        and heading_target is not None
        and heading_target_known is not None
    ):
        h_known_mask = heading_target_known == 1.0
        ax.plot(time_true, np.degrees(heading_true) % 360.0, "k.", ms=1.5, label="True", alpha=0.6)
        ax.plot(
            time_pred,
            np.degrees(pred_heading) % 360.0,
            "r--",
            lw=1.2,
            label="Predicted",
            alpha=0.8,
        )
        ax.plot(
            time_true,
            np.where(h_known_mask, np.degrees(heading_target) % 360.0, np.nan),
            "b-",
            lw=2.0,
            label="Target",
            alpha=0.5,
        )
        ax.set_ylim(-10, 370)
        if (~h_known_mask).any():
            ax.fill_between(
                time_true,
                -10,
                370,
                where=~h_known_mask,
                alpha=0.08,
                color="gray",
                label="Heading target unknown",
            )
        ax.set_ylabel("Heading [°]")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no lateral channel", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Heading [°]")
    ax.grid(True, alpha=0.3)

    # (0,1) Altitude
    ax = axes[0, 1]
    ax.plot(time_true, alt_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(time_pred, pred_alt, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, alt_target, "b-", lw=2.0, label="Target", alpha=0.4)
    _set_ylim(ax, alt_true)
    ax.set_ylabel("Altitude [m]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) TAS
    ax = axes[1, 0]
    ax.plot(time_true, tas_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(time_pred, pred_tas, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, tas_target, "b-", lw=2.0, label="Target", alpha=0.4)
    _set_ylim(ax, tas_true)
    _shade_unknown(ax, time_true, tas_unknown, "TAS unknown")
    ax.set_ylabel("TAS [m/s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) FPA (γ)
    ax = axes[1, 1]
    ax.plot(time_true, np.degrees(gamma_true), "k-", lw=0.8, label="True", alpha=0.5)
    ax.plot(time_pred, np.degrees(pred_gamma), "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, np.degrees(gamma_target), "b-", lw=3.0, label="γ target (known)", alpha=0.9)
    if (gamma_known == 0.0).any():
        ymin = ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -10
        ax.fill_between(
            time_true,
            ymin,
            10,
            where=(gamma_known == 0.0),
            alpha=0.08,
            color="gray",
            label="γ unknown",
        )
    _set_ylim(ax, np.degrees(gamma_true))
    ax.set_ylabel("FPA [°]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,0) Ground track
    axes[2, 0].remove()
    if (
        has_lateral
        and lat_arr is not None
        and lon_arr is not None
        and lat_pred is not None
        and lon_pred is not None
    ):
        if cartopy_available:
            proj = ccrs.PlateCarree()
            ax = fig.add_subplot(4, 2, 5, projection=proj)
            lon_min = float(min(lon_arr.min(), lon_pred.min()))
            lon_max = float(max(lon_arr.max(), lon_pred.max()))
            lat_min = float(min(lat_arr.min(), lat_pred.min()))
            lat_max = float(max(lat_arr.max(), lat_pred.max()))
            lon_margin = max(0.1, 0.10 * (lon_max - lon_min))
            lat_margin = max(0.1, 0.10 * (lat_max - lat_min))
            ax.set_extent(
                [
                    lon_min - lon_margin,
                    lon_max + lon_margin,
                    lat_min - lat_margin,
                    lat_max + lat_margin,
                ],
                crs=proj,
            )
            ax.add_feature(cfeature.OCEAN, facecolor="#e6f0fa", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#f5f0e6", zorder=0)
            ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="0.4", zorder=1)
            ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.6", linestyle=":", zorder=1)
            gl = ax.gridlines(draw_labels=True, lw=0.4, color="0.7", alpha=0.5, zorder=2)
            gl.top_labels = False
            gl.right_labels = False
            straight = ~in_turn_arr if in_turn_arr is not None else np.ones(n, dtype=bool)
            if straight.any():
                ax.plot(
                    lon_arr[straight],
                    lat_arr[straight],
                    ".",
                    color="tab:blue",
                    ms=1.5,
                    label="True (straight)",
                    transform=proj,
                    zorder=3,
                )
            if in_turn_arr is not None and in_turn_arr.any():
                ax.plot(
                    lon_arr[in_turn_arr],
                    lat_arr[in_turn_arr],
                    ".",
                    color="tab:orange",
                    ms=1.5,
                    label="True (in_turn)",
                    transform=proj,
                    zorder=3,
                )
            ax.plot(
                lon_pred,
                lat_pred,
                "r--",
                lw=1.2,
                alpha=0.8,
                label="Predicted",
                transform=proj,
                zorder=4,
            )
            ax.plot(
                lon_pred[-1],
                lat_pred[-1],
                "rv",
                ms=10,
                mfc="none",
                label="end (pred)",
                transform=proj,
                zorder=5,
            )
            ax.plot(lon_arr[0], lat_arr[0], "g^", ms=10, label="start", transform=proj, zorder=5)
            ax.plot(
                lon_arr[-1], lat_arr[-1], "kv", ms=10, label="end (true)", transform=proj, zorder=5
            )
            ax.legend(loc="best", fontsize=7)
        else:
            ax = fig.add_subplot(4, 2, 5)
            ax.plot(lon_arr, lat_arr, "k.", ms=1.5, label="True", alpha=0.6)
            ax.plot(lon_pred, lat_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
            ax.set_xlabel("Longitude [°]")
            ax.set_ylabel("Latitude [°]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    else:
        ax = fig.add_subplot(4, 2, 5)
        ax.text(0.5, 0.5, "no lateral channel", ha="center", va="center", transform=ax.transAxes)

    # (2,1) VZ
    ax = axes[2, 1]
    ax.plot(time_true, vz_true, "k-", lw=1.5, label="True (raw_vz_ms)", alpha=0.8)
    ax.plot(time_pred, vz_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, vz_sel, "b-", lw=2.0, label="VZ target (sel)", alpha=0.4)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    _set_ylim(ax, vz_true)
    _shade_unknown(ax, time_true, vz_unknown, "VZ unknown")
    ax.set_ylabel("VZ [m/s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3,0) CAS
    ax = axes[3, 0]
    ax.plot(time_true, cas_true, "k-", lw=1.5, label="True (bds_ias_ms)", alpha=0.8)
    ax.plot(time_pred, cas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, cas_sel, "b-", lw=2.0, label="CAS target (sel)", alpha=0.4)
    _set_ylim(ax, cas_true)
    _shade_unknown(ax, time_true, cas_unknown, "CAS unknown")
    ax.set_ylabel("CAS [m/s]")
    ax.set_xlabel("Time [min]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3,1) Mach
    ax = axes[3, 1]
    ax.plot(time_true, mach_true, "k-", lw=1.5, label="True (era_mach)", alpha=0.8)
    ax.plot(time_pred, mach_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, mach_sel, "b-", lw=2.0, label="Mach target (sel)", alpha=0.4)
    _set_ylim(ax, mach_true)
    _shade_unknown(ax, time_true, mach_unknown, "Mach unknown")
    ax.set_ylabel("Mach [-]")
    ax.set_xlabel("Time [min]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Neural ODE Inference — {flight_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_visualize(
    *,
    arch: str,
    config: Path,
    typecode: str | None = None,
    flight: str | None = None,
    limit: int | None = None,
) -> None:
    """Generate the inference comparison figure (Node-FDM vs ground truth).

    Reads ground truth from the Delta Table (test split, valid rows) and
    joins per-flight predictions from ``<predict_dir>/<typecode>/<flight_id>.parquet``
    (written by ``fdm predict``).  Produces a 4×2 grid (Heading | Alt,
    TAS | FPA, Ground-track | VZ, CAS | Mach) at
    ``data/figures/inference_<typecode>_<flight_id>.png``.

    Args:
        arch: Architecture identifier (``"qar"`` or ``"adsb"``).
        config: Path to YAML pipeline config.
        typecode: Single typecode to visualize (default: all from config).
        flight: Specific ``meta_flight_id`` to visualize.
        limit: At most N flights per typecode (default: all).
    """
    _require_viz()

    import polars as pl
    from node_fdm_data.delta import read_delta_table

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)

    delta_table = cfg.paths.resolve("delta_table")
    predict_dir = cfg.paths.resolve("predicted_dir")
    figure_dir = cfg.paths.resolve("figure_dir")
    figure_dir.mkdir(parents=True, exist_ok=True)

    typecodes = [typecode] if typecode else cfg.typecodes
    df = read_delta_table(delta_table)
    df = df.filter(pl.col("fdm_flag_valid") & pl.col("meta_split").eq("test"))
    sel_cols = [
        c
        for c in df.columns
        if c.startswith("fdm_") and "_sel" in c and df.schema[c] != pl.Boolean
    ]
    if sel_cols:
        df = df.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols])

    log.info("visualize_start", arch=arch, typecodes=typecodes, rows=len(df))
    written = 0

    for acft in typecodes:
        acft_df = df.filter(pl.col("meta_aircraft_type").eq(acft))
        if len(acft_df) == 0:
            log.warning("visualize_empty_typecode", typecode=acft)
            continue
        flights = acft_df.partition_by("meta_flight_id", maintain_order=True)
        if limit is not None:
            flights = flights[:limit]
        for flight_df in flights:
            flight_id = flight_df["meta_flight_id"][0]
            if flight and flight_id != flight:
                continue
            pred_file = predict_dir / acft / f"{flight_id}.parquet"
            if not pred_file.exists():
                log.warning(
                    "visualize_skip",
                    typecode=acft,
                    flight_id=flight_id,
                    reason="no_predict_parquet",
                )
                continue
            pred_df = pl.read_parquet(pred_file)
            output_path = figure_dir / f"inference_{acft}_{flight_id}.png"
            try:
                _plot_inference_figure(
                    flight_id=flight_id,
                    info=info,
                    flight_df=flight_df.sort("raw_timestamp"),
                    pred_df=pred_df,
                    output_path=output_path,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("visualize_skip", typecode=acft, flight_id=flight_id, error=str(exc))
                continue
            written += 1
            log.info("visualize_saved", path=str(output_path))
            if flight and flight_id == flight:
                break

    log.info("visualize_done", written=written, typecodes=typecodes)


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
        msg = f"performance.parquet not found at {perf_path}. Run 'fdm evaluate' first."
        raise SystemExit(msg)

    _require_viz()

    import altair as alt  # type: ignore[import-not-found,unused-ignore]
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
        return chart  # type: ignore[no-any-return,unused-ignore]

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
        msg = f"example.parquet not found at {example_path}. Run 'fdm visualize' first."
        raise SystemExit(msg)

    _require_viz()

    import altair as alt  # type: ignore[import-not-found,unused-ignore]
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
