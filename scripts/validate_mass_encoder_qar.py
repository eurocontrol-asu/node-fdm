"""Validate the trained MassEncoder against ground-truth QAR mass curves.

The MassEncoder learned on ADS-B (which has no observed mass) predicts the
take-off mass from a 5-feature flight summary. QAR records carry the **true**
mass at 1 Hz (computed from fuel-flow integration onboard), so they are the
gold-standard reference to falsify the encoder.

For each QAR flight, we:

1. Read the parquet, drop ground phases, sort by datetime.
2. Slide a fixed-length window (default 60 s, shift 60 s).
3. At each segment ``t_0``:
   - Read the true mass ``Mass[t_0]`` from QAR.
   - Build the 5 MassEncoder features from the QAR columns
     (``dist_total_flight``, ``dist_adep_at_t0``,
     ``cruise_alt_max_flight``, ``wind_long_mean_flight``,
     ``temp_isa_dev_mean_flight``).
   - Run the MassEncoder.
4. Plot the true mass curve vs. the predicted mass curve, one panel per
   flight, on a single figure.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/validate_mass_encoder_qar.py \\
        --qar-dir /Users/gabriel/Downloads/QAR_EXPORT \\
        --model-name full_hybrid_v2 \\
        --output data/figures/mass_encoder_qar_validation.png

The features are reconstructed from the QAR columns:

  +---------------------------+----------------------------------------------+
  | Feature                   | QAR source                                   |
  +---------------------------+----------------------------------------------+
  | dist_total_flight         | (max - min)(AlongPathDistance) [km → m]      |
  | dist_adep_at_t0           | AlongPathDistance[t_0] - min(...) [km → m]   |
  | cruise_alt_max_flight     | max(Altitude) [ft → m]                       |
  | wind_long_mean_flight     | -mean(TailWind) [m/s]                        |
  | temp_isa_dev_mean_flight  | overridable (default 6.97 K = ADS-B mean)    |
  +---------------------------+----------------------------------------------+

The temperature feature is set to a constant fallback because the QAR_EXPORT
format does not carry ERA5 temperature. The MassEncoder coefficient on this
feature is small (~+0.54), so its contribution to ``m_0`` is bounded and the
validation stays meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog
import torch

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
log = structlog.get_logger("validate_mass_encoder_qar")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FT_TO_M: float = 0.3048
KM_TO_M: float = 1000.0
KT_TO_MS: float = 0.514444
R_AIR: float = 287.05
"""Specific gas constant of dry air (J/(kg·K))."""
GAMMA_AIR: float = 1.4
"""Heat-capacity ratio for dry air."""
A0_MS: float = 340.294
"""Standard sea-level speed of sound (m/s, ISA), used for the qc(CAS) relation."""
P0_PA: float = 101_325.0
"""Standard sea-level pressure (Pa, ISA), used for the qc(CAS) relation."""
DEFAULT_TEMP_ISA_DEV_K: float | None = None
"""Override for ``temp_isa_dev_mean_flight``.

By default (``None``) the script derives the real temperature per row from
TAS/CAS/Altitude (compressible flow inversion) and computes
``mean(T_real − T_ISA(h))``. Pass a number via ``--temp-isa-dev-k`` to
force a constant fallback.
"""

AIR_PHASES: frozenset[str] = frozenset(
    [
        "INI_CLIMB",
        "CLIMB",
        "CRUISE",
        "DESCENT",
        "APPROACH",
        "FIN_APPROACH",
        "TO",
        "LANDING",
    ]
)
"""QAR ``Phase`` labels considered airborne. Ground phases are dropped."""

EARTH_RADIUS_M: float = 6_371_000.0
"""Mean Earth radius (m) used for the haversine cumulative-distance computation."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightFeatures:
    """5-feature row matching ``adsb_hybrid.FLIGHT_FEATURE_COLS`` order."""

    dist_total_flight: float
    dist_adep_at_t0: float
    cruise_alt_max_flight: float
    wind_long_mean_flight: float
    temp_isa_dev_mean_flight: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.dist_total_flight,
                self.dist_adep_at_t0,
                self.cruise_alt_max_flight,
                self.wind_long_mean_flight,
                self.temp_isa_dev_mean_flight,
            ],
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class FlightTrace:
    """Time series of (true mass, predicted m_0) pairs for one QAR flight."""

    name: str
    times_min: np.ndarray
    true_mass_kg: np.ndarray
    pred_mass_kg: np.ndarray
    # Vol-level summary for the figure subtitle
    dist_total_km: float
    cruise_alt_max_ft: float
    wind_long_mean: float
    temp_isa_dev_k: float


# ---------------------------------------------------------------------------
# QAR schema detection (camelCase Airbus vs DOMAIN__SIGNAL Honeywell)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QarSchema:
    """Per-format column mapping used to compute the 5 MassEncoder features.

    Two QAR formats are supported in the wild:

    * **camelCase** (e.g. ``QAR_EXPORT/``): Airbus-style records with
      columns ``Mass``, ``TailWind``, ``Altitude``, ``AlongPathDistance``,
      ``TrueAirSpeed``, ``ComputedAirSpeed``, ``Phase``. Temperature is
      absent → derived from CAS/TAS/altitude via compressible inversion.

    * **DOMAIN__SIGNAL** (Honeywell, e.g. ``QAR2/``): every signal is
      ``DOMAIN__NAME`` (``WEIGHT__GW``, ``TEMP__DELTA_ISA``, ``ALT__STD``,
      ``WIND__LONG``, ``TRAJ__LAT_GPS``, ``TRAJ__LON_GPS``…). Temperature
      deviation is observed directly, distance is reconstructed from GPS
      coordinates via cumulative haversine.

    Detection is done by probing for the presence of ``WEIGHT__GW``.
    """

    name: str
    mass_col: str
    altitude_col: str
    altitude_ft: bool
    wind_long_col: str
    wind_long_sign: float
    delta_isa_col: str | None  # None ⇒ derive via compressible inversion
    has_along_path_km: bool
    along_path_col: str | None
    gps_lat_col: str | None
    gps_lon_col: str | None
    cas_col: str | None
    tas_col: str | None
    phase_col: str | None
    sort_col: str | None


SCHEMA_CAMEL = QarSchema(
    name="camelCase",
    mass_col="Mass",
    altitude_col="Altitude",
    altitude_ft=True,
    wind_long_col="TailWind",
    wind_long_sign=-1.0,  # TailWind sign is opposite to wind_long_mean_flight
    delta_isa_col=None,  # derived from CAS/TAS/Altitude
    has_along_path_km=True,
    along_path_col="AlongPathDistance",
    gps_lat_col=None,
    gps_lon_col=None,
    cas_col="ComputedAirSpeed",
    tas_col="TrueAirSpeed",
    phase_col="Phase",
    sort_col="datetime",
)

SCHEMA_HONEYWELL = QarSchema(
    name="DOMAIN__SIGNAL",
    mass_col="WEIGHT__GW",
    altitude_col="ALT__STD",
    altitude_ft=True,
    wind_long_col="WIND__LONG",
    wind_long_sign=+1.0,  # WIND__LONG is already the longitudinal component
    delta_isa_col="TEMP__DELTA_ISA",
    has_along_path_km=False,
    along_path_col=None,
    gps_lat_col="TRAJ__LAT_GPS",
    gps_lon_col="TRAJ__LON_GPS",
    cas_col="SPD__CAS",
    tas_col="SPD__TAS",
    phase_col=None,  # No standard Phase label — filter on altitude + mass instead
    sort_col=None,
)


def detect_schema(df: pl.DataFrame) -> QarSchema:
    """Pick the right schema by probing for ``WEIGHT__GW``."""
    if "WEIGHT__GW" in df.columns:
        return SCHEMA_HONEYWELL
    if "Mass" in df.columns:
        return SCHEMA_CAMEL
    msg = (
        "Unknown QAR schema: neither 'WEIGHT__GW' (Honeywell) nor 'Mass' "
        "(camelCase) is present in the parquet columns."
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# QAR reading + feature extraction
# ---------------------------------------------------------------------------


def _cas_tas_alt_to_t_real_k(
    cas_ms: np.ndarray,
    tas_ms: np.ndarray,
    p_static_pa: np.ndarray,
) -> np.ndarray:
    """Recover real static temperature from CAS, TAS, and static pressure.

    Uses compressible-flow inversion (subsonic regime):

    1. CAS → impact pressure ``qc`` via the standard subsonic relation:

       .. math::

          q_c = p_0 \\left[ \\left(1 + \\tfrac{\\gamma-1}{2}
          (CAS/a_0)^2 \\right)^{\\gamma/(\\gamma-1)} - 1 \\right]

    2. ``(qc, p_static)`` → Mach number:

       .. math::

          M = \\sqrt{\\tfrac{2}{\\gamma-1}
              \\left[\\left(q_c/p + 1\\right)^{(\\gamma-1)/\\gamma} - 1\\right]}

    3. ``(TAS, M)`` → static temperature:

       .. math::

          T = (TAS / M)^2 / (\\gamma \\cdot R)

    The closed-form qc(CAS) and M(qc, p) account for compressibility, so
    this works up to roughly Mach 1 — well above the operational envelope
    of an A320 (Mach ≤ 0.82 typical).
    """
    g = GAMMA_AIR
    exponent_qc = g / (g - 1.0)
    exponent_m = (g - 1.0) / g
    # 1. qc from CAS
    qc = P0_PA * ((1.0 + 0.5 * (g - 1.0) * (cas_ms / A0_MS) ** 2) ** exponent_qc - 1.0)
    # 2. Mach from qc and static pressure
    mach_sq = (2.0 / (g - 1.0)) * ((qc / p_static_pa + 1.0) ** exponent_m - 1.0)
    mach_sq = np.clip(mach_sq, 1e-6, None)
    mach = np.sqrt(mach_sq)
    # 3. T from TAS and Mach
    return np.asarray((tas_ms / mach) ** 2 / (g * R_AIR), dtype=np.float64)


def _derive_temp_isa_dev_mean_k_compressible(df: pl.DataFrame, schema: QarSchema) -> float:
    """Mean ``T_real − T_ISA(h)`` via compressible CAS/TAS/altitude inversion.

    Used for the camelCase format only, since the Honeywell format already
    carries ``TEMP__DELTA_ISA`` natively.
    """
    from node_fdm_data.physics.isa import isa_pressure, isa_temperature

    assert schema.cas_col is not None
    assert schema.tas_col is not None
    tas_ms = df[schema.tas_col].to_numpy() * KT_TO_MS
    cas_ms = df[schema.cas_col].to_numpy() * KT_TO_MS
    alt_raw = df[schema.altitude_col].to_numpy()
    alt_m = alt_raw * FT_TO_M if schema.altitude_ft else alt_raw

    valid = (cas_ms > 50.0) & (tas_ms > 50.0) & (alt_m > 5_000.0)
    if int(valid.sum()) < 100:
        return 0.0

    tas_v = tas_ms[valid].astype(np.float64)
    cas_v = cas_ms[valid].astype(np.float64)
    alt_v = alt_m[valid].astype(np.float64)

    p_isa = np.asarray(isa_pressure(alt_v), dtype=np.float64)
    t_isa = np.asarray(isa_temperature(alt_v), dtype=np.float64)
    t_real = _cas_tas_alt_to_t_real_k(cas_v, tas_v, p_isa)
    return float(np.nanmean(t_real - t_isa))


def derive_temp_isa_dev_mean_k(df: pl.DataFrame, schema: QarSchema) -> float:
    """Mean ΔT_ISA over the airborne segment, dispatched by schema.

    Honeywell formats expose ``TEMP__DELTA_ISA`` natively; camelCase
    formats fall back to compressible inversion (see helper above).
    """
    if schema.delta_isa_col is not None and schema.delta_isa_col in df.columns:
        return float(df[schema.delta_isa_col].mean())
    return _derive_temp_isa_dev_mean_k_compressible(df, schema)


def _haversine_cumsum_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance (km) from successive GPS points.

    First entry is 0; each subsequent entry adds the haversine arc from the
    previous point. Used to synthesize ``AlongPathDistance`` for the
    Honeywell format which lacks it.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    seg_km = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) / KM_TO_M
    cumsum = np.concatenate([[0.0], np.cumsum(seg_km)])
    return cumsum


def load_qar_flight(path: Path) -> tuple[pl.DataFrame, QarSchema]:
    """Read a QAR parquet, detect schema, drop ground rows, sort by time.

    Returns the cleaned airborne-only frame plus the detected schema.
    """
    df = pl.read_parquet(path)
    schema = detect_schema(df)

    # Drop corrupted rows (mass=0 happens at file boundaries on both formats)
    df = df.filter(
        pl.col(schema.mass_col).is_not_null()
        & (pl.col(schema.mass_col) > 1_000.0)
        & (pl.col(schema.mass_col) < 200_000.0)
    )

    if schema.phase_col is not None and schema.phase_col in df.columns:
        df = df.filter(pl.col(schema.phase_col).is_in(list(AIR_PHASES)))
    else:
        # No Phase column — approximate airborne via altitude (>1 000 ft)
        alt_threshold_raw = 1_000.0 if schema.altitude_ft else 304.8
        df = df.filter(pl.col(schema.altitude_col) > alt_threshold_raw)

    if schema.sort_col is not None and schema.sort_col in df.columns:
        df = df.sort(schema.sort_col)
    elif "FlightTime" in df.columns:
        df = df.sort("FlightTime")
    elif "Time" in df.columns:
        df = df.sort("Time")

    return df, schema


@dataclass(frozen=True)
class FlightAggregates:
    """Vol-level summary needed to assemble per-segment features."""

    dist_total_m: float
    cruise_alt_max_m: float
    wind_long_mean: float
    temp_isa_dev_k: float
    along_path_km: np.ndarray
    """Cumulative distance from the first airborne row, in km (length = n_rows)."""


def compute_flight_aggregates(df: pl.DataFrame, schema: QarSchema) -> FlightAggregates:
    """Compute the vol-level features + per-row along-path distance."""
    if schema.has_along_path_km:
        assert schema.along_path_col is not None
        along_path_km = df[schema.along_path_col].to_numpy().astype(np.float64)
        # Offset so the first airborne row sits at 0
        along_path_km = along_path_km - along_path_km[0]
    else:
        assert schema.gps_lat_col is not None and schema.gps_lon_col is not None
        lat = df[schema.gps_lat_col].to_numpy().astype(np.float64)
        lon = df[schema.gps_lon_col].to_numpy().astype(np.float64)
        along_path_km = _haversine_cumsum_km(lat, lon)

    dist_total_m = float(along_path_km[-1] - along_path_km[0]) * KM_TO_M
    alt_raw = df[schema.altitude_col]
    cruise_alt_max_m = float(alt_raw.max()) * (FT_TO_M if schema.altitude_ft else 1.0)
    wind_long_mean = schema.wind_long_sign * float(df[schema.wind_long_col].mean())
    temp_isa_dev_k = derive_temp_isa_dev_mean_k(df, schema)

    return FlightAggregates(
        dist_total_m=dist_total_m,
        cruise_alt_max_m=cruise_alt_max_m,
        wind_long_mean=wind_long_mean,
        temp_isa_dev_k=temp_isa_dev_k,
        along_path_km=along_path_km,
    )


def build_segment_features(
    along_path_km_t0: float,
    aggregates: FlightAggregates,
    temp_isa_dev_k_override: float | None,
) -> FlightFeatures:
    """Assemble the 5-feature tensor for one segment at instant ``t_0``."""
    return FlightFeatures(
        dist_total_flight=aggregates.dist_total_m,
        dist_adep_at_t0=along_path_km_t0 * KM_TO_M,
        cruise_alt_max_flight=aggregates.cruise_alt_max_m,
        wind_long_mean_flight=aggregates.wind_long_mean,
        temp_isa_dev_mean_flight=(
            temp_isa_dev_k_override
            if temp_isa_dev_k_override is not None
            else aggregates.temp_isa_dev_k
        ),
    )


# ---------------------------------------------------------------------------
# MassEncoder loading
# ---------------------------------------------------------------------------


def load_mass_encoder(
    model_dir: Path,
    feature_cols: Sequence[str],
    oew_kg: float,
    mtow_kg: float,
) -> torch.nn.Module:
    """Re-instantiate ``MassEncoderLinear`` and load saved weights."""
    from node_fdm.layers.mass_encoder import MassEncoderLinear
    from node_fdm_data.schemas.adsb_hybrid import FLIGHT_FEATURE_SIGNS

    meta = json.loads((model_dir / "meta.json").read_text())
    feature_stats = {col: meta["stats_dict"][col] for col in feature_cols}

    encoder = MassEncoderLinear(
        feature_stats=feature_stats,
        feature_cols=list(feature_cols),
        expected_signs=list(FLIGHT_FEATURE_SIGNS),
        oew_kg=oew_kg,
        mtow_kg=mtow_kg,
    )
    state = torch.load(model_dir / "mass_encoder.pt", weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Per-flight processing
# ---------------------------------------------------------------------------


def process_flight(
    path: Path,
    encoder: torch.nn.Module,
    *,
    step_s: int,
    seq_len_s: int,
    temp_isa_dev_k_override: float | None,
) -> FlightTrace | None:
    """Build the segment-level true/predicted mass series for one QAR flight.

    Auto-detects the QAR schema (camelCase Airbus vs DOMAIN__SIGNAL
    Honeywell). When ``temp_isa_dev_k_override`` is ``None`` the value is
    sourced from the dataset itself: directly from ``TEMP__DELTA_ISA`` on
    Honeywell, or derived via compressible inversion on camelCase. Pass an
    explicit value to force a constant fallback (sensitivity analysis).
    """
    df, schema = load_qar_flight(path)
    if df.shape[0] < seq_len_s:
        log.warning("flight_too_short", path=path.name, n_rows=df.shape[0])
        return None

    aggregates = compute_flight_aggregates(df, schema)
    effective_temp_isa_dev_k = (
        temp_isa_dev_k_override
        if temp_isa_dev_k_override is not None
        else aggregates.temp_isa_dev_k
    )
    log.debug(
        "flight_temp_isa_dev",
        path=path.name,
        schema=schema.name,
        temp_isa_dev_k=round(effective_temp_isa_dev_k, 2),
    )

    times_min: list[float] = []
    true_mass_kg: list[float] = []
    pred_mass_kg: list[float] = []

    mass = df[schema.mass_col].to_numpy()
    along_path_km = aggregates.along_path_km
    if "FlightTime" in df.columns:
        flight_time = df["FlightTime"].to_numpy()
    elif schema.sort_col is not None and schema.sort_col in df.columns:
        # datetime → seconds since first row
        ts = df[schema.sort_col].to_numpy().astype("datetime64[s]")
        flight_time = (ts - ts[0]).astype(float)
    else:
        flight_time = np.arange(df.shape[0], dtype=float)

    with torch.no_grad():
        for start in range(0, df.shape[0] - seq_len_s + 1, step_s):
            true_m = float(mass[start])
            if not np.isfinite(true_m) or true_m < 30_000 or true_m > 100_000:
                continue
            along_km = float(along_path_km[start])
            if not np.isfinite(along_km):
                continue
            features = build_segment_features(
                along_path_km_t0=along_km,
                aggregates=aggregates,
                temp_isa_dev_k_override=effective_temp_isa_dev_k,
            )
            pred = encoder(features.to_tensor().unsqueeze(0)).item()
            times_min.append(float(flight_time[start]) / 60.0)
            true_mass_kg.append(true_m)
            pred_mass_kg.append(pred)

    if not times_min:
        log.warning("flight_yielded_no_segments", path=path.name)
        return None

    return FlightTrace(
        name=path.stem,
        times_min=np.asarray(times_min),
        true_mass_kg=np.asarray(true_mass_kg),
        pred_mass_kg=np.asarray(pred_mass_kg),
        dist_total_km=aggregates.dist_total_m / KM_TO_M,
        cruise_alt_max_ft=aggregates.cruise_alt_max_m / FT_TO_M,
        wind_long_mean=aggregates.wind_long_mean,
        temp_isa_dev_k=effective_temp_isa_dev_k,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _grid_dims(n: int) -> tuple[int, int]:
    """Return ``(rows, cols)`` for a near-square grid of ``n`` panels."""
    if n <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def plot_traces(
    traces: Sequence[FlightTrace],
    output_path: Path,
    oew_kg: float,
    mtow_kg: float,
) -> None:
    """Plot all traces on a grid: one panel per flight."""
    if not traces:
        log.error("no_traces_to_plot")
        return

    rows, cols = _grid_dims(len(traces))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.5 * cols, 4.0 * rows),
        squeeze=False,
    )

    for idx, trace in enumerate(traces):
        ax = axes[idx // cols][idx % cols]
        ax.plot(trace.times_min, trace.true_mass_kg, "k-", lw=1.6, label="True (QAR)")
        ax.plot(trace.times_min, trace.pred_mass_kg, "r--", lw=1.4, label="Predicted (MassEncoder)")
        ax.axhline(oew_kg, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(mtow_kg, color="gray", ls=":", lw=0.8, alpha=0.5)
        # Bias on this flight
        bias = float(np.mean(trace.pred_mass_kg - trace.true_mass_kg))
        rmse = float(np.sqrt(np.mean((trace.pred_mass_kg - trace.true_mass_kg) ** 2)))
        ax.set_title(
            f"{trace.name}\n"
            f"dist={trace.dist_total_km:.0f} km · max_alt={trace.cruise_alt_max_ft:.0f} ft · "
            f"wind={trace.wind_long_mean:+.1f} m/s · ΔT_ISA={trace.temp_isa_dev_k:+.1f} K\n"
            f"bias={bias:+.0f} kg · RMSE={rmse:.0f} kg",
            fontsize=9,
        )
        ax.set_xlabel("Flight time [min]")
        ax.set_ylabel("Mass [kg]")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for idx in range(len(traces), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(
        "MassEncoder vs. QAR ground truth (Phase 1 validation)",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("figure_saved", path=str(output_path), n_flights=len(traces))


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def print_summary(traces: Iterable[FlightTrace]) -> None:
    """Print a per-flight bias / RMSE table + overall stats."""
    rows = list(traces)
    if not rows:
        return
    print()
    print("=" * 78)
    print(f"{'Flight':<45s} {'TOW(kg)':>9s} {'pred(kg)':>10s} {'bias(kg)':>9s}")
    print("-" * 78)
    biases: list[float] = []
    rmses: list[float] = []
    for tr in rows:
        true_tow = float(tr.true_mass_kg[0])
        pred_tow = float(tr.pred_mass_kg[0])
        bias = float(np.mean(tr.pred_mass_kg - tr.true_mass_kg))
        rmse = float(np.sqrt(np.mean((tr.pred_mass_kg - tr.true_mass_kg) ** 2)))
        biases.append(bias)
        rmses.append(rmse)
        short = tr.name[:45]
        print(f"{short:<45s} {true_tow:>9.0f} {pred_tow:>10.0f} {bias:>+9.0f}")
    print("-" * 78)
    print(
        f"Mean bias = {np.mean(biases):+.0f} kg   |   "
        f"Mean RMSE = {np.mean(rmses):.0f} kg   |   "
        f"n_flights = {len(rows)}"
    )
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MassEncoder predictions against QAR ground truth.",
    )
    parser.add_argument(
        "--qar-dir",
        type=Path,
        default=Path("/Users/gabriel/Downloads/QAR_EXPORT"),
        help="Directory containing QAR_EXPORT *.parquet files (one per flight).",
    )
    parser.add_argument(
        "--model-name",
        default="full_hybrid_v2",
        help="Sub-directory under data/models/ containing the trained checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Pipeline config YAML (used to resolve models_dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/figures/mass_encoder_qar_validation.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--seq-len-s",
        type=int,
        default=60,
        help="Segment length in seconds (must match training seq_len).",
    )
    parser.add_argument(
        "--step-s",
        type=int,
        default=60,
        help="Step between consecutive segments in seconds.",
    )
    parser.add_argument(
        "--temp-isa-dev-k",
        type=float,
        default=DEFAULT_TEMP_ISA_DEV_K,
        help=(
            "Constant override for 'temp_isa_dev_mean_flight'. When unset "
            "(default), the script derives the value per flight from "
            "TAS/CAS/Altitude via compressible-flow inversion. Pass a "
            "scalar to force a fixed value (sensitivity analysis)."
        ),
    )
    parser.add_argument(
        "--max-flights",
        type=int,
        default=None,
        help="Maximum number of flights to process (default: all).",
    )
    args = parser.parse_args()

    from node_fdm_data.schemas.adsb_hybrid import (
        A320_MTOW_KG,
        A320_OEW_KG,
        FLIGHT_FEATURE_COLS,
    )
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(Path(args.config))
    models_dir = cfg.paths.resolve("models_dir")
    model_dir = models_dir / args.model_name
    if not model_dir.exists():
        log.error("model_dir_missing", path=str(model_dir))
        return 1

    encoder = load_mass_encoder(
        model_dir,
        feature_cols=FLIGHT_FEATURE_COLS,
        oew_kg=A320_OEW_KG,
        mtow_kg=A320_MTOW_KG,
    )
    log.info("mass_encoder_loaded", model_dir=str(model_dir))

    qar_files = sorted(args.qar_dir.glob("*.parquet"))
    if args.max_flights is not None:
        qar_files = qar_files[: args.max_flights]
    if not qar_files:
        log.error("no_qar_files", qar_dir=str(args.qar_dir))
        return 2

    log.info("processing_qar_flights", n_files=len(qar_files))

    traces: list[FlightTrace] = []
    for path in qar_files:
        try:
            trace = process_flight(
                path,
                encoder,
                step_s=args.step_s,
                seq_len_s=args.seq_len_s,
                temp_isa_dev_k_override=args.temp_isa_dev_k,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("process_flight_failed", path=path.name, error=str(exc))
            continue
        if trace is not None:
            traces.append(trace)

    if not traces:
        log.error("no_valid_traces")
        return 3

    plot_traces(traces, args.output, oew_kg=A320_OEW_KG, mtow_kg=A320_MTOW_KG)
    print_summary(traces)

    return 0


if __name__ == "__main__":
    sys.exit(main())
