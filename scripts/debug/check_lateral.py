"""Diagnostic plot: lateral state vs target for one flight.

Mirrors ``check_gamma_target.py`` but for the lateral channel.  Computes
``augment_lateral`` on the fly (the pipeline does not produce these
columns yet) and lays out 6 panels:

1. Ground track (lat/lon) coloured by ``in_turn``
2. Track + heading (raw signals from BDS / GPS)
3. Reference tracks: ``track_ortho`` (great circle to segment end B)
   and ``track_loxo`` (rhumb line A->B), overlaid on actual track
4. Drift (track - heading) and estimated drift
   from ERA5 wind triangle
5. Heading target = track_ortho - drift_est, vs heading actual
6. Diff signals: ``track_diff`` and ``heading_diff`` -- visual sanity
   check that they agree on straight segments (4s step assumption)

Usage:
    uv run python scripts/debug/check_lateral.py [--flight FLIGHT_ID]
                                                  [--delta PATH]
                                                  [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable
from pygeomag import GeoMag

from node_fdm_data.lateral import augment_lateral

# Module-level GeoMag instance: WMM-2025 coefficients shipped with pygeomag.
# Single instance reused across flights (model file load is the bulk of cost).
_GEOMAG = GeoMag()

DEFAULT_FLIGHTS = [
    "738284_ISR826_s0",
    "40666a_EZY78QT_s0",
    "4a0443_BTI7FR_s0",
    "a88774_JBU493_s0",
    "44006e_AUA445_s0",
]

KT_TO_MS = 0.5144444


def _wrap_signed_deg(x: np.ndarray) -> np.ndarray:
    """Wrap to [-180, 180]."""
    return (x + 180.0) % 360.0 - 180.0


def _prepare_for_lateral(flight: pl.DataFrame) -> pl.DataFrame:
    """Rename raw_* / bds_hdg_deg to the names augment_lateral expects."""
    return flight.rename(
        {
            "raw_lat_deg": "latitude",
            "raw_lon_deg": "longitude",
            "raw_track_deg": "track",
            "bds_hdg_deg": "heading",
            "era_tas_kt": "TAS",
        }
    )


def _decimal_year(ts: object) -> float:
    """Convert a datetime-like (UTC) to a decimal year suitable for WMM.

    The WMM models magnetic declination as a slowly varying function of time.
    Using one mid-flight timestamp is sufficient: declination drifts at most
    ~0.1 deg/yr, negligible over a single flight.
    """
    import datetime as _dt

    if isinstance(ts, _dt.datetime):
        dt_obj = ts
    else:  # polars / pandas style
        dt_obj = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=_dt.UTC)
    year = dt_obj.year
    start = _dt.datetime(year, 1, 1, tzinfo=_dt.UTC)
    end = _dt.datetime(year + 1, 1, 1, tzinfo=_dt.UTC)
    return year + (dt_obj - start).total_seconds() / (end - start).total_seconds()


def _declination_series(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_ft: np.ndarray,
    decimal_year: float,
) -> np.ndarray:
    """Per-sample magnetic declination [deg] using WMM (pygeomag).

    pygeomag expects altitude in km. We pass actual altitude (effect is tiny;
    < 0.01 deg between 0 and FL400) but fall back to 0 km for NaN samples.
    """
    n = lat_deg.size
    out = np.full(n, np.nan, dtype=np.float64)
    alt_km = np.where(np.isnan(alt_ft), 0.0, alt_ft * 0.3048 / 1000.0)
    for i in range(n):
        if np.isnan(lat_deg[i]) or np.isnan(lon_deg[i]):
            continue
        try:
            res = _GEOMAG.calculate(
                glat=float(lat_deg[i]),
                glon=float(lon_deg[i]),
                alt=float(alt_km[i]),
                time=decimal_year,
            )
            out[i] = res.d
        except Exception:  # noqa: BLE001 -- diagnostic script, log nothing
            out[i] = np.nan
    return out


def _drift_from_wind(
    heading_deg: np.ndarray,
    tas_ms: np.ndarray,
    u_wind_ms: np.ndarray,
    v_wind_ms: np.ndarray,
) -> np.ndarray:
    """Estimate drift (deg, signed) from ERA5 wind triangle.

    drift = atan2(V_wind_lat, TAS + V_wind_long)
    where _lat is the cross-heading wind component (positive = wind from
    the left, pushes the aircraft right => track > heading).
    """
    psi = np.radians(heading_deg)
    # Project wind onto along/cross-heading basis.
    # Aviation convention: u = east wind, v = north wind.
    # Heading 0 = north, 90 = east. Aircraft body-x points along heading.
    along = u_wind_ms * np.sin(psi) + v_wind_ms * np.cos(psi)
    cross = u_wind_ms * np.cos(psi) - v_wind_ms * np.sin(psi)
    return np.degrees(np.arctan2(cross, tas_ms + along))


def plot_flight(df: pl.DataFrame, flight_id: str, out_dir: Path) -> Path | None:
    flight = df.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
    if flight.height < 20:
        print(f"skip {flight_id}: too short ({flight.height} rows)", file=sys.stderr)
        return None

    flight = _prepare_for_lateral(flight)
    flight = augment_lateral(flight)

    n = flight.height
    t_min = np.arange(n) * 4.0 / 60.0  # 4 s sampling -> minutes

    lat = flight["latitude"].to_numpy()
    lon = flight["longitude"].to_numpy()
    track = flight["track"].to_numpy().astype(np.float64)
    heading = flight["heading"].to_numpy().astype(np.float64)
    in_turn = flight["in_turn"].to_numpy().astype(bool)
    track_ortho = flight["track_ortho"].to_numpy().astype(np.float64)

    tas_ms = flight["TAS"].to_numpy().astype(np.float64) * KT_TO_MS
    u_wind = flight["era_u_wind_ms"].to_numpy().astype(np.float64)
    v_wind = flight["era_v_wind_ms"].to_numpy().astype(np.float64)
    drift_wind = _drift_from_wind(heading, tas_ms, u_wind, v_wind)

    # Heading target = track_target - drift (option A, projection in heading frame)
    heading_target = _wrap_signed_deg(track_ortho - drift_wind)
    heading_target_mod = heading_target % 360.0

    # Diff signals (signed, [-180, 180])
    track_diff = _wrap_signed_deg(track_ortho - track)
    heading_diff = _wrap_signed_deg(heading_target - heading)

    # ---------------------------------------------------------------- magnetic declination
    # Pull a single mid-flight timestamp; declination is essentially constant
    # over one flight (~0.1 deg/yr), so per-sample time is unnecessary.
    ts_series = flight["raw_timestamp"]
    mid_ts = ts_series[ts_series.len() // 2]
    decimal_year = _decimal_year(mid_ts)

    alt_ft = flight["raw_alt_ft"].to_numpy().astype(np.float64)
    declination = _declination_series(lat, lon, alt_ft, decimal_year)
    bds_true = _wrap_signed_deg(heading + declination) % 360.0
    track_minus_drift = _wrap_signed_deg(track - drift_wind) % 360.0

    # ---------------------------------------------------------------- plot
    fig = plt.figure(figsize=(16, 21))
    gs = fig.add_gridspec(7, 1, height_ratios=[2.2, 1, 1, 1, 1, 1, 1], hspace=0.40)

    meta = flight.row(0, named=True)
    fig.suptitle(
        f"Lateral state vs target -- {flight_id}  "
        f"({meta.get('meta_departure', '?')} -> {meta.get('meta_arrival', '?')})",
        fontsize=13,
        fontweight="bold",
    )

    # 1) Ground track
    ax = fig.add_subplot(gs[0])
    straight = ~in_turn
    ax.plot(lon[straight], lat[straight], ".", color="tab:blue", ms=2, label="straight")
    ax.plot(lon[in_turn], lat[in_turn], ".", color="tab:orange", ms=2, label="in_turn")
    ax.plot(lon[0], lat[0], "g^", ms=10, label="start")
    ax.plot(lon[-1], lat[-1], "rv", ms=10, label="end")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Ground track (turn detection)", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # 2) Track + heading raw
    ax = fig.add_subplot(gs[1])
    ax.plot(t_min, track, color="tab:blue", lw=0.8, alpha=0.8, label="track (GPS)")
    ax.plot(t_min, heading, color="tab:purple", lw=0.8, alpha=0.8, label="heading (BDS magnetic)")
    _shade_turns(ax, t_min, in_turn)
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Raw signals: track (sol) vs heading (cap)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) Track + reference track (current -> B great circle)
    ax = fig.add_subplot(gs[2])
    ax.plot(t_min, track, color="tab:blue", lw=0.6, alpha=0.6, label="track actual")
    ax.plot(t_min, track_ortho, color="tab:red", lw=1.5, alpha=0.9, label="track_ortho (target)")
    _shade_turns(ax, t_min, in_turn)
    ax.set_ylabel("Track [deg]")
    ax.set_title("Track actual vs reference track (NaN outside enclosing segment)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Drift: geometric (track - heading) vs wind-triangle estimate
    ax = fig.add_subplot(gs[3])
    drift_geom_signed = _wrap_signed_deg(track - heading)
    ax.plot(
        t_min,
        drift_geom_signed,
        color="tab:gray",
        lw=0.6,
        alpha=0.6,
        label="track - heading (raw)",
    )
    ax.plot(
        t_min,
        drift_wind,
        color="tab:orange",
        lw=1.2,
        alpha=0.9,
        label="drift estimated from ERA5 wind",
    )
    ax.axhline(0, color="black", lw=0.5, ls="--")
    _shade_turns(ax, t_min, in_turn)
    ax.set_ylabel("Drift [deg]")
    ax.set_title(
        "Drift comparison -- raw track-heading includes magnetic declination (~+/-5 deg)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5) Heading target vs heading actual
    ax = fig.add_subplot(gs[4])
    ax.plot(t_min, heading, color="tab:purple", lw=0.7, alpha=0.7, label="heading actual")
    ax.plot(
        t_min,
        heading_target_mod,
        color="tab:red",
        lw=1.5,
        alpha=0.9,
        label="heading target = track_ortho - drift_est",
    )
    _shade_turns(ax, t_min, in_turn)
    ax.set_ylabel("Heading [deg]")
    ax.set_title("Projected heading target (option A) vs heading", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6) Diff signals -- the question: track_diff approx heading_diff ?
    ax = fig.add_subplot(gs[5])
    ax.plot(t_min, track_diff, color="tab:blue", lw=1.0, alpha=0.8, label="track_diff")
    ax.plot(
        t_min,
        heading_diff,
        color="tab:red",
        lw=1.0,
        alpha=0.8,
        ls="--",
        label="heading_diff",
    )
    ax.axhline(0, color="black", lw=0.5, ls="--")
    _shade_turns(ax, t_min, in_turn)
    ax.set_ylabel("Diff [deg]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Error signals fed to ODE -- compare track_diff vs heading_diff", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7) Heading reconciliation: bds_true vs track_minus_drift vs heading_target
    ax = fig.add_subplot(gs[6])
    ax.plot(
        t_min,
        bds_true,
        color="tab:purple",
        lw=1.0,
        alpha=0.85,
        label="bds_hdg + declination (BDS true)",
    )
    ax.plot(
        t_min,
        track_minus_drift,
        color="tab:orange",
        lw=1.0,
        alpha=0.85,
        ls="--",
        label="track - drift_ERA5 (track-derived true)",
    )
    ax.plot(
        t_min,
        heading_target_mod,
        color="tab:red",
        lw=0.8,
        alpha=0.5,
        label="heading_target (track_ortho - drift_est)",
    )
    _shade_turns(ax, t_min, in_turn)
    ax.set_ylabel("Heading [deg]")
    ax.set_xlabel("Time [min]")
    ax.set_title(
        "Heading reconciliation -- BDS+declination vs track-drift (both true frame)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{flight_id}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Quick numeric sanity check on straight segments
    valid = straight & ~np.isnan(track_diff) & ~np.isnan(heading_diff)
    if valid.any():
        diff_of_diffs = np.abs(track_diff[valid] - heading_diff[valid])
        msg = (
            f"wrote {out_path}  "
            f"|track_diff - heading_diff| on straight: "
            f"med={np.median(diff_of_diffs):.2f} deg, "
            f"p95={np.percentile(diff_of_diffs, 95):.2f} deg"
        )
    else:
        msg = f"wrote {out_path}  (no valid straight points)"

    # New: |bds_true - track_minus_drift| on straight segments
    recon_diff = np.abs(_wrap_signed_deg(bds_true - track_minus_drift))
    valid_recon = (
        straight
        & ~np.isnan(bds_true)
        & ~np.isnan(track_minus_drift)
        & ~np.isnan(declination)
    )
    if valid_recon.any():
        rd = recon_diff[valid_recon]
        decl_med = np.nanmedian(declination[valid_recon])
        msg += (
            f"  |  |bds_true - track_minus_drift| straight: "
            f"med={np.median(rd):.2f} deg, p95={np.percentile(rd, 95):.2f} deg "
            f"(n={valid_recon.sum()}, decl_med={decl_med:.2f} deg)"
        )
    else:
        msg += "  |  reconciliation: no valid straight points"
    print(msg)
    return out_path


def _shade_turns(ax: plt.Axes, t_min: np.ndarray, in_turn: np.ndarray) -> None:
    """Light grey shading on turn intervals."""
    if not in_turn.any():
        return
    edges = np.diff(in_turn.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for s, e in zip(starts, ends, strict=True):
        ax.axvspan(t_min[s], t_min[min(e, len(t_min) - 1)], color="black", alpha=0.06)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flight",
        action="append",
        help="Flight ID to plot (repeatable). Defaults to 5 reference flights.",
    )
    parser.add_argument(
        "--delta",
        type=Path,
        default=Path("data/flights.delta"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/figures/lateral"),
    )
    args = parser.parse_args()

    if not args.delta.exists():
        print(f"Delta table not found: {args.delta}", file=sys.stderr)
        raise SystemExit(1)

    dt = DeltaTable(str(args.delta))
    df = pl.DataFrame(dt.to_pyarrow_table())

    flights = args.flight or DEFAULT_FLIGHTS
    for fid in flights:
        plot_flight(df, fid, args.out_dir)


if __name__ == "__main__":
    main()
