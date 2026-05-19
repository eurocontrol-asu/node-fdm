"""Helper for building train/validation datasets from a Delta Table DataFrame.

Reads flight data grouped by ``meta_flight_id``, windows them into
sequences, and returns typed :class:`FlightDataset` instances.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog
import torch

from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm_data.physics.isa import isa_temperature
from node_fdm_data.schemas.adsb_hybrid import FLIGHT_FEATURE_COLS_6 as FLIGHT_FEATURE_COLS

# Physical clamp range for ``fdm_mach_sel`` when building the flight-level
# aggregate ``mach_cruise_planned``. A320 cruise Mach is typically 0.78-0.80;
# values outside [0.4, 0.85] are spurious plateau detections (climb residue,
# noise on bds_mach decoded outliers).
_MACH_CRUISE_MIN: float = 0.4
_MACH_CRUISE_MAX: float = 0.85
# Fallback used when no usable Mach selection exists in the flight. Set to
# the empirical A320 mean from the dataset audit (~0.78).
_MACH_CRUISE_FALLBACK: float = 0.78

__all__ = [
    "FLIGHT_FEATURE_COLS",
    "get_train_val_data",
]

log = structlog.get_logger("node_fdm.loader")


def _fill_nan_sel(df: pl.DataFrame) -> pl.DataFrame:
    """Fill NaN and null to 0.0 on numeric ``fdm_*_sel*`` columns.

    Boolean companions like ``fdm_track_sel_known`` are skipped — they
    carry presence info, not a value, and ``fill_nan`` is unsupported
    on bool dtype.
    """
    sel_cols = [
        c for c in df.columns if c.startswith("fdm_") and "_sel" in c and df.schema[c].is_numeric()
    ]
    if sel_cols:
        df = df.with_columns(
            [pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols],
        )
    return df


def _load_and_window(
    flights_df: pl.DataFrame,
    x_cols: list[str],
    u_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
    seq_len: int,
    shift: int,
    *,
    flight_limit: int | None = None,
    e1_cols: list[str] | None = None,
    flight_feature_cols: list[str] | None = None,
    require_routing: bool = False,
) -> list[FlightSample]:
    """Group flights and slice into fixed-length windows.

    Args:
        flights_df: DataFrame with all rows for one split, containing
            ``meta_flight_id`` and optionally ``fdm_flag_distance_ok``.
        x_cols: State column names.
        u_cols: Control column names.
        e_cols: Environment column names.
        dx_cols: Derivative column names.
        seq_len: Window length.
        shift: Step between windows.
        flight_limit: Max number of flights to process.
        e1_cols: Optional extra environment column names.
        flight_feature_cols: Optional flight-level feature names to attach.
        require_routing: Whether to drop flights with no routing data.

    Returns:
        List of windowed :class:`FlightSample` instances.
    """
    samples: list[FlightSample] = []
    requested_flight_feature_cols = flight_feature_cols or []
    has_flight_features = bool(requested_flight_feature_cols)
    _require_routing = require_routing or has_flight_features
    routing_cols = ["fdm_adep_dist_m", "fdm_ades_dist_m"]
    aggregate_source_cols = ["raw_alt_m", "fdm_long_wind_ms", "era_temp_K"]
    feature_source_cols = routing_cols + aggregate_source_cols if has_flight_features else []

    # Synthetic state/derivative columns are emitted at runtime (by the
    # MassEncoder, not observed in the delta). Skip them from the existence
    # check and zero-pad their slot when assembling the per-window tensors.
    synthetic_cols: set[str] = {"fdm_mass_kg", "fdm_d_mass_kgs"}
    real_x_cols = [c for c in x_cols if c not in synthetic_cols]
    real_dx_cols = [c for c in dx_cols if c not in synthetic_cols]
    synthetic_x_idx = [i for i, c in enumerate(x_cols) if c in synthetic_cols]
    synthetic_dx_idx = [i for i, c in enumerate(dx_cols) if c in synthetic_cols]

    all_cols = real_x_cols + u_cols + e_cols + real_dx_cols + feature_source_cols

    # Verify all columns exist
    missing = [c for c in all_cols if c not in flights_df.columns]
    if missing:
        log.warning("missing_columns", missing=missing)
        return samples

    # Resolve valid E1 columns (skip missing with warning)
    valid_e1_cols: list[str] = []
    if e1_cols:
        for col in e1_cols:
            if col in flights_df.columns:
                valid_e1_cols.append(col)
            else:
                log.warning("e1_column_missing", column=col)

    has_distance_flag = "fdm_flag_distance_ok" in flights_df.columns
    has_weight_col = "fdm_train_weight" in flights_df.columns
    flights_dropped_no_routing = 0
    flight_aggregates: dict[object, dict[str, float]] = {}

    flight_ids = flights_df.get_column("meta_flight_id").unique().sort().to_list()
    if flight_limit is not None:
        flight_ids = flight_ids[:flight_limit]

    for fid in flight_ids:
        df = flights_df.filter(pl.col("meta_flight_id") == fid)
        if _require_routing:
            routing_presence = df.select(
                [
                    pl.col("fdm_adep_dist_m").is_not_null().any().alias("has_adep"),
                    pl.col("fdm_ades_dist_m").is_not_null().any().alias("has_ades"),
                ]
            ).row(0, named=True)
            if not routing_presence["has_adep"] and not routing_presence["has_ades"]:
                flights_dropped_no_routing += 1
                continue

        n_rows = len(df)
        if n_rows < seq_len:
            continue

        if has_flight_features:
            alt_arr = df.get_column("raw_alt_m").to_numpy().astype(np.float32)
            temp_arr = df.get_column("era_temp_K").to_numpy().astype(np.float32)
            wind_arr = df.get_column("fdm_long_wind_ms").to_numpy().astype(np.float32)
            temp_isa_dev = temp_arr - isa_temperature(alt_arr)
            # Mach cruise planned: highest FMS-selected Mach observed during
            # the flight, clamped to physically plausible cruise range to
            # discard spurious plateau detections. Falls back to the typical
            # A320 cruise Mach when no usable selection exists.
            mach_cruise_planned = _MACH_CRUISE_FALLBACK
            if "fdm_mach_sel" in df.columns:
                mach_arr = df.get_column("fdm_mach_sel").to_numpy()
                in_range = np.isfinite(mach_arr) & (
                    (mach_arr >= _MACH_CRUISE_MIN) & (mach_arr <= _MACH_CRUISE_MAX)
                )
                if in_range.any():
                    mach_cruise_planned = float(np.max(mach_arr[in_range]))
            flight_aggregates[fid] = {
                "cruise_alt_max_flight": float(np.nanmax(alt_arr)),
                "wind_long_mean_flight": float(np.nanmean(wind_arr)),
                "temp_isa_dev_mean_flight": float(np.nanmean(temp_isa_dev)),
                "mach_cruise_planned": mach_cruise_planned,
            }

        # Extract arrays (empty col lists → zero-width arrays).
        # Synthetic columns (e.g. fdm_mass_kg) are filled by the runtime
        # encoder, not observed; we zero-pad the slot so the tensor shape
        # matches len(x_cols).
        x_arr_real = (
            df.select(real_x_cols).to_numpy().astype(np.float32)
            if real_x_cols
            else np.empty((n_rows, 0), dtype=np.float32)
        )
        if synthetic_x_idx:
            x_arr = np.zeros((n_rows, len(x_cols)), dtype=np.float32)
            real_idx = [i for i in range(len(x_cols)) if i not in synthetic_x_idx]
            x_arr[:, real_idx] = x_arr_real
        else:
            x_arr = x_arr_real

        u_arr = (
            df.select(u_cols).to_numpy().astype(np.float32)
            if u_cols
            else np.empty((n_rows, 0), dtype=np.float32)
        )
        e_arr = df.select(e_cols).to_numpy().astype(np.float32)

        dx_arr_real = (
            df.select(real_dx_cols).to_numpy().astype(np.float32)
            if real_dx_cols
            else np.empty((n_rows, 0), dtype=np.float32)
        )
        if synthetic_dx_idx:
            dx_arr = np.zeros((n_rows, len(dx_cols)), dtype=np.float32)
            real_idx = [i for i in range(len(dx_cols)) if i not in synthetic_dx_idx]
            dx_arr[:, real_idx] = dx_arr_real
        else:
            dx_arr = dx_arr_real

        e1_arr: np.ndarray | None = None
        if valid_e1_cols:
            e1_arr = df.select(valid_e1_cols).to_numpy().astype(np.float32)

        # Distance flag array for segment filtering (AC6)
        dist_ok: np.ndarray | None = None
        if has_distance_flag:
            dist_ok = df.get_column("fdm_flag_distance_ok").to_numpy()

        w_arr: np.ndarray | None = None
        if has_weight_col:
            w_arr = df.get_column("fdm_train_weight").to_numpy().astype(np.float32)

        adep_arr: np.ndarray | None = None
        ades_arr: np.ndarray | None = None
        # Read routing arrays as soon as either:
        #   - flight features are requested (segment-level features need them), or
        #   - require_routing is set (we filter windows on t_0 presence so the
        #     baseline and hybrid runs end up on the *same* sample set — fair
        #     comparison requires apples-to-apples window selection, not just
        #     apples-to-apples flight selection).
        if (
            _require_routing
            and "fdm_adep_dist_m" in df.columns
            and "fdm_ades_dist_m" in df.columns
        ):
            adep_arr = df.get_column("fdm_adep_dist_m").to_numpy().astype(np.float32)
            ades_arr = df.get_column("fdm_ades_dist_m").to_numpy().astype(np.float32)

        for start in range(0, n_rows - seq_len + 1, shift):
            end = start + seq_len

            # Check for NaN / inf (u_arr excluded: gamma_target NaN
            # is expected and handled by TrajectoryLayer).
            # Lateral channel (Phase 2B): ``fdm_heading_rad`` is in X_COLS
            # and is NaN where ``fdm_heading_known=False`` (no BDS source +
            # unreliable wind triangle).  Such windows are dropped silently
            # here — same mechanism the longitudinal channel relies on for
            # ``fdm_gamma_rad``.  We deliberately do NOT fill NaN→0 on x:
            # injecting a false zero heading would corrupt the integrated
            # state without any way to recover the true value.
            slices = [
                x_arr[start:end],
                e_arr[start:end],
                dx_arr[start:end],
            ]
            if not all(np.isfinite(s).all() for s in slices):
                continue

            # Also check e1 columns for NaN / inf
            if e1_arr is not None and not np.isfinite(e1_arr[start:end]).all():
                continue

            # Segment filter: all rows in window must have distance_ok (AC6)
            if dist_ok is not None and not dist_ok[start:end].all():
                continue

            # Window-level routing guard: drop windows whose t_0 row lacks
            # ADEP / ADES distance. Applied whenever require_routing is on
            # (regardless of flight_feature_cols) so baseline and hybrid
            # share the same sample set for fair comparison (§9.1).
            adep_at_t0: float | None = None
            ades_at_t0: float | None = None
            if _require_routing:
                if adep_arr is None or ades_arr is None:
                    continue
                adep_at_t0 = float(adep_arr[start])
                ades_at_t0 = float(ades_arr[start])
                if not np.isfinite(adep_at_t0) or not np.isfinite(ades_at_t0):
                    continue

            e1_tensor: torch.Tensor | None = None
            if e1_arr is not None:
                e1_tensor = torch.from_numpy(e1_arr[start:end].copy())

            w_tensor: torch.Tensor | None = None
            if w_arr is not None:
                w_tensor = torch.from_numpy(w_arr[start:end].copy())

            flight_features_tensor: torch.Tensor | None = None
            if has_flight_features:
                # adep_at_t0 / ades_at_t0 are already validated by the
                # require_routing guard above (has_flight_features auto-
                # promotes _require_routing).
                assert adep_at_t0 is not None and ades_at_t0 is not None
                feature_values = {
                    "dist_total_flight": adep_at_t0 + ades_at_t0,
                    "dist_adep_at_t0": adep_at_t0,
                    **flight_aggregates[fid],
                }
                feature_row = [feature_values[col] for col in requested_flight_feature_cols]
                flight_features_tensor = (
                    torch.tensor(feature_row, dtype=torch.float32).unsqueeze(0).expand(seq_len, -1)
                )

            samples.append(
                FlightSample(
                    x=torch.from_numpy(x_arr[start:end].copy()),
                    u=torch.from_numpy(u_arr[start:end].copy()),
                    e=torch.from_numpy(e_arr[start:end].copy()),
                    dx=torch.from_numpy(dx_arr[start:end].copy()),
                    e1=e1_tensor,
                    w=w_tensor,
                    flight_features=flight_features_tensor,
                )
            )

    if _require_routing:
        log.info("flights_dropped_no_routing", count=flights_dropped_no_routing)

    return samples


def get_train_val_data(
    data_df: pl.DataFrame,
    x_cols: list[str],
    u_cols: list[str],
    e_cols: list[str],
    dx_cols: list[str],
    *,
    seq_len: int = 60,
    shift: int = 60,
    train_limit: int | None = None,
    val_limit: int | None = None,
    e1_cols: list[str] | None = None,
    flight_feature_cols: list[str] | None = None,
    require_routing: bool = False,
) -> tuple[FlightDataset, FlightDataset]:
    """Create training and validation datasets from Delta Table data.

    The input DataFrame must contain ``meta_split`` and ``meta_flight_id``
    columns. Rows should already be filtered on ``fdm_flag_valid``.

    Fills NaN/null to 0.0 on ``fdm_*_sel*`` columns (U columns) at load
    time, so no separate preprocessing step is needed.

    Args:
        data_df: DataFrame loaded from the Delta Table, filtered on
            ``fdm_flag_valid`` and (optionally) ``meta_aircraft_type``.
        x_cols: State column names (SI units).
        u_cols: Control column names (SI units).
        e_cols: Environment column names (SI units).
        dx_cols: Derivative column names (SI units).
        seq_len: Window length for each sample.
        shift: Step between consecutive windows.
        train_limit: Max number of training flights to load.
        val_limit: Max number of validation flights to load.
        flight_feature_cols: Optional flight-level feature names to attach.
        require_routing: Whether to drop flights with no routing data.

    Returns:
        Tuple of ``(train_dataset, val_dataset)``.
    """
    # Fill NaN→0.0 on _sel columns (AC3)
    data_df = _fill_nan_sel(data_df)

    # Split by meta_split (AC2)
    train_df = data_df.filter(pl.col("meta_split") == "train")
    val_df = data_df.filter(pl.col("meta_split") == "val")

    n_train_flights = train_df.get_column("meta_flight_id").n_unique()
    n_val_flights = val_df.get_column("meta_flight_id").n_unique()
    log.info("loading_data", train_flights=n_train_flights, val_flights=n_val_flights)

    train_samples = _load_and_window(
        train_df,
        x_cols,
        u_cols,
        e_cols,
        dx_cols,
        seq_len=seq_len,
        shift=shift,
        flight_limit=train_limit,
        e1_cols=e1_cols,
        flight_feature_cols=flight_feature_cols,
        require_routing=require_routing,
    )
    val_samples = _load_and_window(
        val_df,
        x_cols,
        u_cols,
        e_cols,
        dx_cols,
        seq_len=seq_len,
        shift=shift,
        flight_limit=val_limit,
        e1_cols=e1_cols,
        flight_feature_cols=flight_feature_cols,
        require_routing=require_routing,
    )

    if flight_feature_cols or require_routing:
        routing_df = data_df.select(
            [
                pl.col("meta_flight_id"),
                pl.col("fdm_adep_dist_m").is_not_null().over("meta_flight_id").alias("has_adep"),
                pl.col("fdm_ades_dist_m").is_not_null().over("meta_flight_id").alias("has_ades"),
            ]
        )
        routing_summary = routing_df.group_by("meta_flight_id").agg(
            [
                pl.col("has_adep").any(),
                pl.col("has_ades").any(),
            ]
        )
        n_flights_with_routing = routing_summary.filter(
            pl.col("has_adep") | pl.col("has_ades")
        ).height
        n_flights_dropped = routing_summary.height - n_flights_with_routing
        n_samples_with_features = sum(
            sample.flight_features is not None for sample in train_samples + val_samples
        )
        log.info(
            "flight_features_loaded",
            n_flights_with_routing=n_flights_with_routing,
            n_flights_dropped=n_flights_dropped,
            n_samples_with_features=n_samples_with_features,
        )

    log.info(
        "data_loaded",
        train_samples=len(train_samples),
        val_samples=len(val_samples),
    )

    return FlightDataset(train_samples), FlightDataset(val_samples)
