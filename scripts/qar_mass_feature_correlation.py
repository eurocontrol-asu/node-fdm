"""Point-wise mass correlation analysis on QAR data.

For each QAR flight, we slide a fixed-length window and at every segment
boundary ``t_0`` extract:

* The **target**: instantaneous mass ``Mass(t_0)`` (or ``WEIGHT__GW(t_0)``).
* A large bank of **candidate features**, both raw QAR columns and
  engineered combinations (cumulative distances, time since takeoff,
  fuel-flow integrals, atmospheric proxies, ...).

We then compute three correlation metrics between the target and every
candidate, on the **pooled** sample (across all flights):

1. **Pearson** — linear monotonic correlation.
2. **Spearman** — rank correlation (catches monotonic non-linear).
3. **Mutual information** — distribution-free dependence (catches non-monotonic).

The script prints a ranked table and, optionally, plots a scatter matrix
of the top-K features against the mass.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/qar_mass_feature_correlation.py \\
        --qar-dir /Users/gabriel/Downloads/QAR3 \\
        --output data/figures/qar_mass_feature_correlation.png

Designed to be **format-agnostic**: handles both the camelCase (Airbus)
and Honeywell (``DOMAIN__SIGNAL``) schemas via the same detection helper
used by ``validate_mass_encoder_qar.py``.

It is intentionally a research tool — the output is meant to inform the
choice of MassEncoder features for the next phase, not to run in CI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

# Reuse the schema detection + loader from the validation script so both
# stay in sync.
sys.path.insert(0, str(Path(__file__).parent))
from validate_mass_encoder_qar import (  # noqa: E402
    EARTH_RADIUS_M,
    FT_TO_M,
    KM_TO_M,
    KT_TO_MS,
    QarSchema,
    SCHEMA_CAMEL,
    SCHEMA_HONEYWELL,
    _haversine_cumsum_km,
    detect_schema,
)

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
log = structlog.get_logger("qar_mass_feature_correlation")


# ---------------------------------------------------------------------------
# Per-segment feature extraction
# ---------------------------------------------------------------------------


# Mapping: candidate feature name → (Honeywell column, camelCase column or None)
# When a column is missing the value is computed on the fly via _ENGINEERED.
_RAW_COLUMN_MAP: dict[str, tuple[str | None, str | None]] = {
    "altitude_m": ("ALT__STD", "Altitude"),
    "mach": ("SPD__MACH", None),
    "cas_ms": ("SPD__CAS", "ComputedAirSpeed"),
    "tas_ms": ("SPD__TAS", "TrueAirSpeed"),
    "ground_speed_ms": ("SPD__GND", "GroundSpeed"),
    "vertical_speed_ms": ("ATT__VV", "VerticalSpeed"),
    "fpa_rad": (None, "gamma"),
    "tailwind_ms": ("WIND__TAIL_WIND", "TailWind"),
    "longwind_ms": ("WIND__LONG", None),
    "crosswind_ms": ("WIND__CROSS_WIND", None),
    "wind_speed_ms": ("WIND__SPD", None),
    "delta_isa_k": ("TEMP__DELTA_ISA", None),
    "sat_c": ("TEMP__SAT", None),
    "n1_left": ("ENG__N1_LEFT", "EngineLeft"),
    "n1_right": ("ENG__N1_RIGHT", "EngineRight"),
    "fuel_flow_left_kgs": (None, "FuelFlowLeft"),
    "fuel_flow_right_kgs": (None, "FuelFlowRight"),
    "fuel_flow_total_kgs": (None, "FuelFlow"),
    "flaps_angle": ("CTL__FLAP_LEVER", "FlapsAngle"),
    "lat_deg": ("TRAJ__LAT_GPS", "Latitude"),
    "lon_deg": ("TRAJ__LON_GPS", "Longitude"),
}


@dataclass(frozen=True)
class FeatureRow:
    """All features extracted at one segment ``t_0`` for one flight."""

    target_mass_kg: float
    raw: dict[str, float]
    engineered: dict[str, float]


def _read_first_value(df: pl.DataFrame, column: str | None) -> np.ndarray | None:
    """Read a column as float array, returning None for missing or non-numeric."""
    if column is None or column not in df.columns:
        return None
    series = df[column]
    if not series.dtype.is_numeric():
        return None
    return series.to_numpy()


def extract_raw_features(
    df: pl.DataFrame,
    schema: QarSchema,
) -> dict[str, np.ndarray]:
    """Read every raw candidate feature available in this QAR frame."""
    out: dict[str, np.ndarray] = {}
    use_honeywell = schema is SCHEMA_HONEYWELL
    for feature_name, (honeywell_col, camel_col) in _RAW_COLUMN_MAP.items():
        col = honeywell_col if use_honeywell else camel_col
        arr = _read_first_value(df, col)
        if arr is None:
            continue
        out[feature_name] = arr.astype(np.float64)

    # Normalize altitude to meters
    if "altitude_m" in out and schema.altitude_ft:
        out["altitude_m"] = out["altitude_m"] * FT_TO_M
    # camelCase TAS/CAS are knots — convert to m/s
    if not use_honeywell:
        for k in ("cas_ms", "tas_ms", "ground_speed_ms", "vertical_speed_ms"):
            if k in out:
                # ground_speed and vertical_speed in QAR camel are also knots / ft/min,
                # but we keep them as-is for now: correlations are scale-invariant.
                pass
        # Convert TAS/CAS to m/s for downstream q_dynamic etc.
        for k in ("cas_ms", "tas_ms"):
            if k in out:
                out[k] = out[k] * KT_TO_MS

    return out


def _is_cruise_mask(df: pl.DataFrame, schema: QarSchema, alt_m: np.ndarray) -> np.ndarray:
    """Boolean mask identifying cruise rows.

    Honeywell QARs lack a ``Phase`` column, so we approximate via altitude
    (above 25 000 ft → cruise regime). On camelCase QARs we honour the
    ``Phase == "CRUISE"`` label directly.
    """
    if schema.phase_col is not None and schema.phase_col in df.columns:
        return df[schema.phase_col].to_numpy() == "CRUISE"
    return alt_m > 25_000.0 * FT_TO_M


def compute_engineered_features(
    raw: dict[str, np.ndarray],
    df: pl.DataFrame,
    schema: QarSchema,
) -> dict[str, np.ndarray]:
    """Derive higher-level features from raw QAR signals.

    All features are arrays of length ``n_rows`` so we can index at every
    segment ``t_0``.

    Two flavours of features are produced:

    1. **Causal in-flight** features (``*_so_far``, ``flight_time_min``,
       ``dist_from_origin_km``…): values evolve along the segment and can
       be evaluated at any ``t_0``.

    2. **Pre-flight features** (``*_planned``, ``dist_total_*``): constant
       per flight, simulating what an OFP/FMS would expose **before
       pushback**. On QAR we approximate them by aggregating the cruise
       phase ex-post — the dispatcher's plan and the actual cruise are
       close enough to use the QAR aggregate as a proxy.
    """
    n_rows = df.shape[0]
    out: dict[str, np.ndarray] = {}

    # --- Cumulative distance from origin ---------------------------------
    if "lat_deg" in raw and "lon_deg" in raw:
        cum_km = _haversine_cumsum_km(raw["lat_deg"], raw["lon_deg"])
        out["dist_from_origin_km"] = cum_km
        out["dist_to_destination_km"] = cum_km[-1] - cum_km

    # --- Time in flight ---------------------------------------------------
    if "FlightTime" in df.columns:
        ft = df["FlightTime"].to_numpy().astype(np.float64)
        out["flight_time_min"] = (ft - ft[0]) / 60.0
    elif schema.sort_col and schema.sort_col in df.columns:
        ts = df[schema.sort_col].to_numpy().astype("datetime64[s]").astype(np.int64)
        out["flight_time_min"] = (ts - ts[0]).astype(np.float64) / 60.0
    else:
        out["flight_time_min"] = np.arange(n_rows, dtype=np.float64) / 60.0

    # --- Cumulative fuel burned -----------------------------------------
    ff = None
    for k in ("fuel_flow_total_kgs", "fuel_flow_left_kgs", "fuel_flow_right_kgs"):
        if k in raw:
            ff = raw.get("fuel_flow_total_kgs")
            break
    if "fuel_flow_left_kgs" in raw and "fuel_flow_right_kgs" in raw:
        ff = raw["fuel_flow_left_kgs"] + raw["fuel_flow_right_kgs"]
    if ff is not None:
        # FuelFlow in camelCase QAR is kg/s (verified by magnitudes ~0.5-1).
        # Honeywell variant unknown — we just integrate the array as-is.
        dt = np.gradient(out["flight_time_min"]) * 60.0  # seconds
        out["cum_fuel_burned_kg"] = np.cumsum(ff * dt)

    # --- Dynamic pressure proxy ------------------------------------------
    if "tas_ms" in raw and "altitude_m" in raw:
        # rho ~ rho_0 * exp(-h/8000) is a crude scale-height approximation
        rho = 1.225 * np.exp(-raw["altitude_m"] / 8000.0)
        out["q_dynamic_pa_proxy"] = 0.5 * rho * raw["tas_ms"] ** 2

    # --- TAS/CAS ratio (compressibility proxy / mass-independent) --------
    if "tas_ms" in raw and "cas_ms" in raw:
        mask = raw["cas_ms"] > 1.0
        ratio = np.where(mask, raw["tas_ms"] / np.maximum(raw["cas_ms"], 1e-6), np.nan)
        out["tas_over_cas"] = ratio

    # --- N1 mean / asymmetry --------------------------------------------
    if "n1_left" in raw and "n1_right" in raw:
        out["n1_mean"] = 0.5 * (raw["n1_left"] + raw["n1_right"])
        out["n1_asymmetry_abs"] = np.abs(raw["n1_left"] - raw["n1_right"])

    # --- Climb rate proxy (altitude derivative) --------------------------
    if "altitude_m" in raw:
        out["climb_rate_ms_proxy"] = np.gradient(raw["altitude_m"])

    # --- Rolling means up to t_0 (causal aggregates) ---------------------
    if "altitude_m" in raw:
        out["altitude_cum_mean_so_far"] = np.cumsum(raw["altitude_m"]) / np.arange(
            1, n_rows + 1
        )
    if "tailwind_ms" in raw:
        out["tailwind_cum_mean_so_far"] = np.cumsum(raw["tailwind_ms"]) / np.arange(
            1, n_rows + 1
        )
    if "delta_isa_k" in raw:
        out["delta_isa_cum_mean_so_far"] = np.cumsum(raw["delta_isa_k"]) / np.arange(
            1, n_rows + 1
        )

    # --- Max altitude seen so far (causal cumulative max) ----------------
    if "altitude_m" in raw:
        out["altitude_max_so_far"] = np.maximum.accumulate(raw["altitude_m"])

    # ---------------------------------------------------------------------
    # Pre-flight features (constant per flight, proxies for what an
    # OFP/FMS would expose before pushback). Computed once from cruise
    # aggregates and broadcast to every row so they can be sampled at any
    # t_0 alongside the causal features.
    # ---------------------------------------------------------------------
    if "altitude_m" in raw:
        cruise_mask = _is_cruise_mask(df, schema, raw["altitude_m"])
        cruise_count = int(cruise_mask.sum())
    else:
        cruise_mask = np.zeros(n_rows, dtype=bool)
        cruise_count = 0

    def _broadcast(value: float) -> np.ndarray:
        return np.full(n_rows, value, dtype=np.float64)

    # Total route distance (pre-flight): max along-path distance
    if "dist_from_origin_km" in out:
        out["dist_total_flight_km_PRE"] = _broadcast(float(out["dist_from_origin_km"][-1]))

    # Cruise altitude planned (pre-flight): mean altitude during cruise
    # phase, or fall back to max altitude when cruise mask is empty.
    if "altitude_m" in raw:
        if cruise_count >= 50:
            cruise_alt_mean_m = float(np.nanmean(raw["altitude_m"][cruise_mask]))
        else:
            cruise_alt_mean_m = float(np.nanmax(raw["altitude_m"]))
        out["cruise_alt_planned_m_PRE"] = _broadcast(cruise_alt_mean_m)
        out["cruise_alt_max_flight_m_PRE"] = _broadcast(float(np.nanmax(raw["altitude_m"])))

    # Cruise Mach planned (pre-flight): mean Mach during cruise
    if "mach" in raw and cruise_count >= 50:
        out["mach_cruise_planned_PRE"] = _broadcast(
            float(np.nanmean(raw["mach"][cruise_mask]))
        )

    # Cruise TAS planned (pre-flight): mean TAS during cruise — informative
    # without a Mach column on camelCase QARs.
    if "tas_ms" in raw and cruise_count >= 50:
        out["tas_cruise_planned_ms_PRE"] = _broadcast(
            float(np.nanmean(raw["tas_ms"][cruise_mask]))
        )

    # Forecast wind component (pre-flight): mean longitudinal wind on route
    if "longwind_ms" in raw:
        out["wind_long_mean_flight_PRE"] = _broadcast(
            float(np.nanmean(raw["longwind_ms"]))
        )
    elif "tailwind_ms" in raw:
        out["wind_long_mean_flight_PRE"] = _broadcast(
            -float(np.nanmean(raw["tailwind_ms"]))
        )

    # Forecast temperature deviation (pre-flight): mean ΔT_ISA over route
    if "delta_isa_k" in raw:
        out["temp_isa_dev_mean_flight_PRE"] = _broadcast(
            float(np.nanmean(raw["delta_isa_k"]))
        )

    return out


def sample_flight_features(
    path: Path,
    *,
    seq_len_s: int,
    step_s: int,
) -> list[FeatureRow] | None:
    """Slide windows over a QAR flight and return one FeatureRow per ``t_0``."""
    from validate_mass_encoder_qar import load_qar_flight

    df, schema = load_qar_flight(path)
    if df.shape[0] < seq_len_s:
        log.warning("flight_too_short", path=path.name, n_rows=df.shape[0])
        return None

    raw_arrays = extract_raw_features(df, schema)
    eng_arrays = compute_engineered_features(raw_arrays, df, schema)
    mass = df[schema.mass_col].to_numpy().astype(np.float64)

    rows: list[FeatureRow] = []
    for start in range(0, df.shape[0] - seq_len_s + 1, step_s):
        m = float(mass[start])
        if not np.isfinite(m) or m < 30_000 or m > 100_000:
            continue
        raw_t0 = {
            k: float(arr[start]) for k, arr in raw_arrays.items() if np.isfinite(arr[start])
        }
        eng_t0 = {
            k: float(arr[start]) for k, arr in eng_arrays.items() if np.isfinite(arr[start])
        }
        rows.append(FeatureRow(target_mass_kg=m, raw=raw_t0, engineered=eng_t0))
    return rows


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationStats:
    feature: str
    n_samples: int
    pearson: float
    spearman: float
    mutual_info: float
    kind: str  # "raw" | "engineered"


def _pooled_features(rows: Sequence[FeatureRow]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pool every FeatureRow into ``(mass_array, {feature: array_with_nans})``."""
    keys: set[str] = set()
    kinds: dict[str, str] = {}
    for r in rows:
        for k in r.raw:
            keys.add(k)
            kinds.setdefault(k, "raw")
        for k in r.engineered:
            keys.add(k)
            kinds.setdefault(k, "engineered")

    n = len(rows)
    mass = np.empty(n, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {k: np.full(n, np.nan, dtype=np.float64) for k in keys}
    for i, r in enumerate(rows):
        mass[i] = r.target_mass_kg
        for k, v in r.raw.items():
            arrays[k][i] = v
        for k, v in r.engineered.items():
            arrays[k][i] = v
    return mass, arrays


def compute_correlation_table(
    rows: Sequence[FeatureRow],
    *,
    mi_n_neighbors: int = 5,
) -> list[CorrelationStats]:
    """Return Pearson / Spearman / MI for every feature vs. target mass."""
    mass, arrays = _pooled_features(rows)
    stats: list[CorrelationStats] = []
    raw_keys = set()
    for r in rows:
        raw_keys.update(r.raw)
    for feature, values in arrays.items():
        finite = np.isfinite(values) & np.isfinite(mass)
        if int(finite.sum()) < 50:
            continue
        x = values[finite]
        y = mass[finite]
        # Skip constant features
        if np.std(x) < 1e-9:
            continue
        try:
            pearson_r = float(pearsonr(x, y).statistic)
            spearman_r = float(spearmanr(x, y).statistic)
            mi = float(
                mutual_info_regression(
                    x.reshape(-1, 1), y, n_neighbors=mi_n_neighbors, random_state=0
                )[0]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("corr_failed", feature=feature, error=str(exc))
            continue
        stats.append(
            CorrelationStats(
                feature=feature,
                n_samples=int(finite.sum()),
                pearson=pearson_r,
                spearman=spearman_r,
                mutual_info=mi,
                kind="raw" if feature in raw_keys else "engineered",
            )
        )
    # Rank by |Spearman| (catches monotonic non-linear)
    stats.sort(key=lambda s: abs(s.spearman), reverse=True)
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_correlation_table(stats: Sequence[CorrelationStats]) -> None:
    print()
    print("=" * 92)
    print(
        f"{'Feature':<32s} {'kind':<11s} {'n':>6s} {'Pearson':>9s} "
        f"{'Spearman':>10s} {'MI(nats)':>10s}"
    )
    print("-" * 92)
    for s in stats:
        print(
            f"{s.feature:<32s} {s.kind:<11s} {s.n_samples:>6d} "
            f"{s.pearson:>+9.3f} {s.spearman:>+10.3f} {s.mutual_info:>10.3f}"
        )
    print("=" * 92)


def plot_top_k_scatter(
    rows: Sequence[FeatureRow],
    stats: Sequence[CorrelationStats],
    output_path: Path,
    *,
    top_k: int = 9,
) -> None:
    """Plot ``top_k`` features as scatter (feature, mass) ranked by |Spearman|."""
    if not stats:
        return
    top = stats[:top_k]
    mass, arrays = _pooled_features(rows)
    rows_grid = int(np.ceil(np.sqrt(top_k)))
    cols_grid = int(np.ceil(top_k / rows_grid))
    fig, axes = plt.subplots(
        rows_grid,
        cols_grid,
        figsize=(4.5 * cols_grid, 3.8 * rows_grid),
        squeeze=False,
    )
    for idx, s in enumerate(top):
        ax = axes[idx // cols_grid][idx % cols_grid]
        x = arrays[s.feature]
        finite = np.isfinite(x) & np.isfinite(mass)
        ax.scatter(x[finite], mass[finite], s=3, alpha=0.25, c="tab:blue")
        ax.set_xlabel(s.feature)
        ax.set_ylabel("Mass [kg]")
        ax.set_title(
            f"#{idx + 1} {s.feature}\n"
            f"ρ={s.spearman:+.3f} · r={s.pearson:+.3f} · MI={s.mutual_info:.2f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
    # Hide unused panels
    for idx in range(len(top), rows_grid * cols_grid):
        axes[idx // cols_grid][idx % cols_grid].axis("off")
    fig.suptitle(
        f"Top-{top_k} features correlated with instantaneous mass (pooled QAR segments)",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("scatter_saved", path=str(output_path), top_k=top_k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate QAR features with instantaneous mass.",
    )
    parser.add_argument(
        "--qar-dir",
        type=Path,
        default=Path("/Users/gabriel/Downloads/QAR3"),
        help="Directory of QAR .parquet files (Honeywell or camelCase).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/figures/qar_mass_feature_correlation.png"),
        help="Output figure path (top-K scatter matrix).",
    )
    parser.add_argument(
        "--seq-len-s",
        type=int,
        default=60,
        help="Segment length in seconds.",
    )
    parser.add_argument(
        "--step-s",
        type=int,
        default=60,
        help="Step between consecutive segments.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many features to plot.",
    )
    args = parser.parse_args()

    qar_files = sorted(args.qar_dir.glob("*.parquet"))
    if not qar_files:
        log.error("no_qar_files", qar_dir=str(args.qar_dir))
        return 1
    log.info("scanning", n_files=len(qar_files))

    all_rows: list[FeatureRow] = []
    for path in qar_files:
        rows = sample_flight_features(path, seq_len_s=args.seq_len_s, step_s=args.step_s)
        if rows:
            all_rows.extend(rows)
            log.info("flight_sampled", path=path.name, n_segments=len(rows))

    if not all_rows:
        log.error("no_segments_collected")
        return 2

    log.info("pooling", n_segments=len(all_rows))
    stats = compute_correlation_table(all_rows)
    print_correlation_table(stats)
    plot_top_k_scatter(all_rows, stats, args.output, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
