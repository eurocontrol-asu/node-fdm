"""Plot all Delta Table parameters grouped by physical quantity, one flight per run."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from deltalake import DeltaTable

# ---------------------------------------------------------------------------
# Parameter groups: (group_title, unit, [(column, label, style), ...])
# ---------------------------------------------------------------------------
GROUPS: list[tuple[str, str, list[tuple[str, str, dict[str, object]]]]] = [
    (
        "Altitude",
        "ft",
        [
            ("raw_alt_ft", "raw alt", {"alpha": 0.4, "lw": 0.5}),
            ("bds_mcp_sel_alt_ft", "MCP sel alt", {"ls": "--", "lw": 1.2}),
            ("fdm_alt_sel_ft", "fdm alt sel", {"ls": "-", "lw": 1.5, "color": "tab:red"}),
            ("fdm_mcp_alt_sel_ft", "fdm MCP alt sel", {"ls": ":", "lw": 1.2}),
        ],
    ),
    (
        "TAS",
        "kt",
        [
            ("raw_gs_kt", "GS raw", {"alpha": 0.3, "lw": 0.5, "color": "tab:gray"}),
            ("bds_tas_kt", "BDS TAS raw", {"alpha": 0.4, "lw": 0.6}),
            ("bds_tas_kt_clean", "BDS TAS clean", {"alpha": 0.7, "lw": 0.9}),
            ("era_tas_kt", "ERA5 TAS", {"alpha": 0.6, "lw": 1.0, "color": "tab:orange"}),
            (
                "bds_tas_from_cas_kt",
                "TAS from CAS clean",
                {"lw": 1.5, "color": "tab:green"},
            ),
            (
                "fdm_tas_target_kt",
                "fdm TAS target",
                {"lw": 1.5, "color": "tab:red"},
            ),
        ],
    ),
    (
        "CAS / IAS",
        "kt",
        [
            ("bds_ias_kt", "BDS IAS raw", {"alpha": 0.4, "lw": 0.6}),
            ("bds_ias_kt_clean", "BDS IAS clean", {"alpha": 0.7, "lw": 0.9}),
            ("era_cas_kt", "ERA5 CAS", {"alpha": 0.6, "lw": 1.0, "color": "tab:orange"}),
            ("fdm_cas_sel_kt", "fdm CAS sel", {"lw": 1.5, "color": "tab:red"}),
            (
                "fdm_cas_target_kt",
                "fdm CAS target",
                {"lw": 1.2, "ls": ":", "color": "tab:red"},
            ),
        ],
    ),
    (
        "Mach",
        "",
        [
            ("bds_mach", "BDS mach raw", {"alpha": 0.4, "lw": 0.6}),
            ("bds_mach_clean", "BDS mach clean", {"alpha": 0.7, "lw": 0.9}),
            ("era_mach", "ERA5 mach", {"alpha": 0.6, "lw": 1.0, "color": "tab:orange"}),
            ("fdm_mach_sel", "fdm mach sel", {"lw": 1.5, "color": "tab:red"}),
        ],
    ),
    (
        "Vertical speed",
        "ft/min",
        [
            ("raw_vz_ftmin", "raw Vz", {"alpha": 0.4, "lw": 0.5}),
            ("fdm_vz_sel_ftmin", "fdm Vz sel", {"lw": 1.5, "color": "tab:red"}),
        ],
    ),
    (
        "Flight-path angle",
        "rad",
        [
            ("fdm_gamma_rad", "gamma", {"alpha": 0.5, "lw": 0.8}),
            ("fdm_gamma_sel_rad", "gamma sel", {"lw": 1.5, "color": "tab:red"}),
        ],
    ),
    (
        "Heading / Track",
        "deg",
        [
            ("raw_track_deg", "track", {"alpha": 0.4, "lw": 0.5}),
            ("bds_hdg_deg", "BDS heading", {"alpha": 0.6, "lw": 0.8}),
        ],
    ),
    (
        "Wind",
        "m/s",
        [
            ("era_u_wind_ms", "u wind", {"lw": 1.0}),
            ("era_v_wind_ms", "v wind", {"lw": 1.0}),
            ("fdm_long_wind_ms", "long wind", {"lw": 1.2, "color": "tab:red"}),
        ],
    ),
    (
        "Temperature",
        "K",
        [
            ("era_temp_K", "ERA5 temp", {"lw": 1.2}),
        ],
    ),
    (
        "Distance",
        "nm",
        [
            ("fdm_adep_dist_nm", "dist ADEP", {"lw": 1.0}),
            ("fdm_ades_dist_nm", "dist ADES", {"lw": 1.0}),
        ],
    ),
    (
        "Derivatives",
        "",
        [
            ("fdm_d_tas_ms2", "d(TAS)/dt [m/s²]", {"lw": 0.8}),
            ("fdm_d_vz_ms", "d(Vz)/dt [m/s²]", {"lw": 0.8}),
            ("fdm_d_gamma_rads", "d(gamma)/dt [rad/s]", {"lw": 0.8}),
        ],
    ),
    (
        "Flags",
        "",
        [
            ("fdm_flag_valid", "valid", {"lw": 1.0, "alpha": 0.7}),
            ("fdm_flag_min_speed", "min speed", {"lw": 0.8, "ls": "--", "alpha": 0.5}),
            ("fdm_flag_distance_ok", "distance ok", {"lw": 0.8, "ls": ":", "alpha": 0.5}),
        ],
    ),
]


def plot_flight(df: pl.DataFrame, flight_id: str) -> None:
    """Plot all parameter groups for a single flight."""
    flight = df.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
    ts = flight["raw_timestamp"].to_list()

    # Filter groups that have at least one column present
    active_groups = []
    for title, unit, cols in GROUPS:
        present = [(c, lbl, style) for c, lbl, style in cols if c in flight.columns]
        if present:
            active_groups.append((title, unit, present))

    n = len(active_groups)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    meta = flight.row(0, named=True)
    fig.suptitle(
        f"{flight_id}  —  {meta.get('meta_departure', '?')} → {meta.get('meta_arrival', '?')}"
        f"  ({meta.get('meta_aircraft_type', '?')})",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (title, unit, cols) in zip(axes, active_groups, strict=False):
        for col_name, label, style in cols:
            series = flight[col_name]
            if series.dtype == pl.Boolean:
                vals = series.cast(pl.Int8).to_list()
            else:
                vals = series.to_list()
            ax.plot(ts, vals, label=label, **style)
        ylabel = f"{title} [{unit}]" if unit else title
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=8, ncol=min(len(cols), 4))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (UTC)")
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.show()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    delta_path = repo_root / "data" / "flights.delta"

    if not delta_path.exists():
        print(f"Delta table not found: {delta_path}", file=sys.stderr)
        raise SystemExit(1)

    dt = DeltaTable(str(delta_path))
    df = pl.DataFrame(dt.to_pyarrow_table())

    flights: list[str] = df.get_column("meta_flight_id").unique().sort().to_list()  # type: ignore[assignment]

    if len(sys.argv) > 1:
        # Plot specific flight
        fid = sys.argv[1]
        if fid not in flights:
            print(f"Flight '{fid}' not found. Available: {flights}", file=sys.stderr)
            raise SystemExit(1)
        plot_flight(df, fid)  # type: ignore[arg-type]
    else:
        # Interactive: plot each flight, press Enter for next
        print(f"Found {len(flights)} flights. Press Enter for next, 'q' to quit.")
        for fid in flights:
            print(f"\n→ {fid}")
            plot_flight(df, fid)  # type: ignore[arg-type]
            resp = input("Next? [Enter/q] ")
            if resp.strip().lower() == "q":
                break


if __name__ == "__main__":
    main()
