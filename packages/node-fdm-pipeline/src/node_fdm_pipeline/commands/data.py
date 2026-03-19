"""Data pipeline commands -- aircraft-list, download, preprocess, process.

These commands convert scripts 01-04 from the OpenSky pipeline into typed,
structured CLI functions.  Commands requiring the ``traffic`` library
(aircraft-list, download, preprocess) guard the import and print a helpful
install message if missing.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import polars as pl
    from traffic.core import Flight, Traffic

__all__ = [
    "aircraft_list",
    "download",
    "identify",
    "preprocess",
    "process",
]

log = structlog.get_logger()

# A320 fleet: maximum physically plausible Mach number.
# ADS-B groundspeed outliers can inflate recomputed TAS → Mach > 1.05.
MACH_UPPER = 1.05

# v3 pipeline — column rename mappings (OpenSky raw → raw_*/bds_* convention)
_RAW_RENAME: dict[str, str] = {
    "timestamp": "raw_timestamp",
    "icao24": "raw_icao24",
    "callsign": "raw_callsign",
    "latitude": "raw_lat_deg",
    "longitude": "raw_lon_deg",
    "altitude": "raw_alt_ft",
    "groundspeed": "raw_gs_kt",
    "track": "raw_track_deg",
    "vertical_rate": "raw_vz_ftmin",
}

_BDS_RENAME: dict[str, str] = {
    "selected_mcp": "bds_mcp_sel_alt_ft",
    "selected_fms": "bds_fms_sel_alt_ft",
    "IAS": "bds_ias_kt",
    "TAS": "bds_tas_kt",
    "Mach": "bds_mach",
    "heading": "bds_hdg_deg",
}


# ---------------------------------------------------------------------------
# Traffic guard
# ---------------------------------------------------------------------------


def _require_traffic() -> None:
    """Raise ``SystemExit`` if the ``traffic`` library is not installed."""
    try:
        import traffic  # noqa: F401
    except ImportError:
        print(  # noqa: T201
            "Error: 'traffic' is required for this command.\n"
            "Install with: pip install node-fdm-pipeline[traffic]",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


# ---------------------------------------------------------------------------
# Command 1 — aircraft-list  (script 01)
# ---------------------------------------------------------------------------


def aircraft_list(
    *,
    config: Path,
    sample_size: int = 100,
    query_date: str = "2025-10-01",
    dry_run: bool = False,
) -> None:
    """Query OpenSky for aircraft database and save to CSV.

    Args:
        config: Path to the YAML config file.
        sample_size: Max flights to sample per typecode.
        query_date: Date to query for flight list (YYYY-MM-DD).
        dry_run: Validate config without performing I/O.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("aircraft_list_start", typecodes=cfg.typecodes, query_date=query_date)

    if dry_run:
        log.info("aircraft_list_dry_run", msg="Config valid, would query OpenSky")
        return

    import polars as pl

    _require_traffic()
    from traffic.data import aircraft, opensky

    # Query OpenSky for one day of flights
    next_date = (datetime.strptime(query_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    opensky.trino_client.connect()
    fl_pd = opensky.flightlist(query_date, next_date)
    fl = pl.from_pandas(fl_pd)
    acft_db = pl.from_pandas(aircraft.data[["icao24", "registration", "typecode", "age"]])

    # Join + extract airline code
    ext = acft_db.join(fl, on="icao24", how="inner").with_columns(
        airline=pl.col("callsign").str.slice(0, 3).str.strip_chars()
    )

    # Sample up to N flights per typecode
    sampled = (
        ext.filter(pl.col("typecode").is_in(cfg.typecodes))
        .group_by("typecode")
        .map_groups(lambda g: g.sample(n=min(sample_size, len(g)), seed=42))
    )

    aircraft_db = sampled.select("icao24", "registration", "typecode", "age", "airline")
    output = data_dir / "aircraft_db.csv"
    aircraft_db.write_csv(output)
    log.info("aircraft_list_done", rows=len(aircraft_db), output=str(output))


# ---------------------------------------------------------------------------
# Command 2 — download  (script 02 → v3 étape 0)
# ---------------------------------------------------------------------------


def _rename_to_v3(df: pl.DataFrame, *, batch_date: str) -> pl.DataFrame:
    """Rename raw OpenSky columns to v3 ``raw_*`` / ``bds_*`` convention.

    Args:
        df: DataFrame with original OpenSky column names.
        batch_date: Batch date string (YYYYMMDD) for partitioning.

    Returns:
        DataFrame with ``raw_*`` / ``bds_*`` columns and ``meta_batch_date``.
    """
    import polars as pl

    rename_map = {
        old: new for old, new in {**_RAW_RENAME, **_BDS_RENAME}.items() if old in df.columns
    }
    df = df.rename(rename_map)

    # Keep only raw_*, bds_* columns
    keep = [c for c in df.columns if c.startswith(("raw_", "bds_"))]
    df = df.select(keep)

    return df.with_columns(pl.lit(batch_date).alias("meta_batch_date"))


def download(  # noqa: PLR0912, PLR0915
    *,
    config: Path,
    start_date: str,
    end_date: str,
    step_hours: int = 24,
    dry_run: bool = False,
) -> None:
    """Download ADS-B history and EHS data, write to Delta Table.

    Fetches ADS-B history and Extended Mode-S (EHS) data from OpenSky,
    decodes BDS parameters, renames columns to the v3 ``raw_*`` / ``bds_*``
    convention, and writes the result to a Delta Table partitioned by
    ``meta_batch_date``.

    Args:
        config: Path to the YAML config file.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        step_hours: Hours between download windows.
        dry_run: Validate config without performing I/O.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)

    # Load aircraft list for ICAO24 filtering
    aircraft_csv = cfg.paths.data_dir / "aircraft_db.csv"
    if not aircraft_csv.exists():
        log.error("download_missing_aircraft_db", path=str(aircraft_csv))
        raise SystemExit(
            f"aircraft_db.csv not found at {aircraft_csv}. Run 'fdm aircraft-list' first."
        )

    log.info(
        "download_start",
        start_date=start_date,
        end_date=end_date,
    )

    if dry_run:
        log.info("download_dry_run", msg="Config valid, would download to Delta table")
        return

    import polars as pl
    from node_fdm_data.delta import write_columns

    _require_traffic()
    from traffic.core import Traffic
    from traffic.data import opensky

    delta_table = cfg.paths.resolve("delta_table")
    download_dir = cfg.paths.resolve("download_dir")
    download_dir.mkdir(parents=True, exist_ok=True)

    icao24_list = pl.read_csv(aircraft_csv)["icao24"].to_list()

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    step = timedelta(hours=step_hours)

    all_frames: list[pl.DataFrame] = []

    current = start
    while current < end:
        date_str = current.strftime("%Y%m%d")
        next_day = current + timedelta(hours=24)

        log.info("download_fetch", date=date_str)

        # Fetch ADS-B history
        history = opensky.history(current, next_day, icao24=icao24_list)
        if history is None:
            log.warning("download_empty", kind="history", date=date_str)
            current += step
            continue

        # Fetch raw EHS messages and decode BDS
        extended = opensky.extended(current, next_day, icao24=icao24_list)
        if extended is not None:
            ext_pd = extended.data if hasattr(extended, "data") else extended
            decoder = _RawEHSDecoder(ext_pd)
            decoded_flights: list[Flight] = []
            for flight in history:
                decoded = decoder(flight)
                if decoded is not None:
                    decoded_flights.append(decoded)

            if decoded_flights:
                merged = Traffic.from_flights(decoded_flights)
                df = pl.from_pandas(merged.data)
            else:
                df = pl.from_pandas(history.data)
        else:
            df = pl.from_pandas(history.data)

        # Rename to v3 schema and collect
        df = _rename_to_v3(df, batch_date=date_str)
        all_frames.append(df)
        log.info("download_processed", date=date_str, rows=len(df))

        # Save flightlist for étape 1 (identify)
        flightlist = opensky.flightlist(current, next_day, icao24=icao24_list)
        if flightlist is not None:
            fl_path = download_dir / f"flightlist_{date_str}.parquet"
            if hasattr(flightlist, "to_parquet"):
                flightlist.to_parquet(fl_path)

        current += step

    if all_frames:
        combined = pl.concat(all_frames, how="diagonal_relaxed")
        write_columns(combined, delta_table)
        log.info("download_done", table=str(delta_table), rows=len(combined))


# ---------------------------------------------------------------------------
# Command 2b — identify  (v3 étape 1)
# ---------------------------------------------------------------------------


def _assign_flight_ids(df: pl.DataFrame, gap_threshold_s: int) -> pl.DataFrame:
    """Add meta_original_flight_id and meta_flight_id columns via gap segmentation.

    Args:
        df: Raw Delta Table DataFrame with raw_icao24, raw_callsign, raw_timestamp.
        gap_threshold_s: Gap threshold in seconds for segment splitting.

    Returns:
        DataFrame with _callsign, meta_original_flight_id, meta_flight_id added,
        and temporary columns _dt_s, _seg_global, _seg_idx included for later drop.
    """
    import polars as pl

    df = df.with_columns(
        pl.col("raw_callsign").fill_null("NOCALL").alias("_callsign"),
    )
    df = df.with_columns(
        (pl.col("raw_icao24") + "_" + pl.col("_callsign")).alias("meta_original_flight_id"),
    )
    df = df.sort("raw_icao24", "_callsign", "raw_timestamp")
    df = df.with_columns(
        pl.col("raw_timestamp")
        .diff()
        .dt.total_seconds()
        .over("meta_original_flight_id")
        .alias("_dt_s"),
    )
    df = df.with_columns(
        (pl.col("_dt_s").is_null() | (pl.col("_dt_s") > gap_threshold_s))
        .cum_sum()
        .over("meta_original_flight_id")
        .alias("_seg_global"),
    )
    df = df.with_columns(
        (
            pl.col("_seg_global") - pl.col("_seg_global").min().over("meta_original_flight_id")
        ).alias("_seg_idx"),
    )
    return df.with_columns(
        (pl.col("meta_original_flight_id") + "_s" + pl.col("_seg_idx").cast(pl.Utf8)).alias(
            "meta_flight_id"
        ),
    )


def _join_flightlist(df: pl.DataFrame, download_dir: Path) -> pl.DataFrame:
    """Left-join flightlist metadata onto the flight DataFrame.

    Reads all ``flightlist_*.parquet`` files from *download_dir*, concatenates
    them, and joins departure, arrival, and typecode onto *df*.  Missing
    columns are filled with ``null``.

    Args:
        df: DataFrame with _callsign and raw_icao24 columns.
        download_dir: Directory containing flightlist parquet files.

    Returns:
        DataFrame with meta_departure, meta_arrival, meta_aircraft_type columns.
    """
    import polars as pl

    fl_frames: list[pl.DataFrame] = []
    if download_dir.exists():
        for fl_file in sorted(download_dir.glob("flightlist_*.parquet")):
            fl_frames.append(pl.read_parquet(fl_file))

    if fl_frames:
        fl = pl.concat(fl_frames, how="diagonal_relaxed")
        if len(fl) > 0 and "icao24" in fl.columns:
            if "callsign" in fl.columns:
                fl = fl.with_columns(
                    pl.col("callsign").fill_null("NOCALL").str.strip_chars().alias("callsign"),
                )
            fl_cols = [c for c in ("departure", "arrival", "typecode") if c in fl.columns]
            if fl_cols:
                fl_select = fl.select(["icao24", "callsign", *fl_cols]).unique(
                    subset=["icao24", "callsign"],
                    keep="first",
                )
                df = df.join(
                    fl_select,
                    left_on=["raw_icao24", "_callsign"],
                    right_on=["icao24", "callsign"],
                    how="left",
                )
                rename = {
                    c: f"meta_{alias}"
                    for c, alias in [
                        ("departure", "departure"),
                        ("arrival", "arrival"),
                        ("typecode", "aircraft_type"),
                    ]
                    if c in df.columns
                }
                df = df.rename(rename)

    for col in ("meta_departure", "meta_arrival", "meta_aircraft_type"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
    return df


def identify(
    *,
    config: Path,
    gap_threshold_s: int = 30,
    dry_run: bool = False,
) -> None:
    """Identify flights: segment at gaps, assign flight IDs, join flightlist.

    Reads the Delta Table produced by ``download``, detects temporal gaps
    within each (icao24, callsign) group, assigns ``meta_flight_id`` with
    segment suffixes, and joins flightlist metadata (departure, arrival,
    aircraft type).

    Short segments are **not** filtered — they are flagged at étape 2.

    Args:
        config: Path to the YAML config file.
        gap_threshold_s: Gap threshold in seconds for segment splitting.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_data.delta import read_delta_table, write_columns

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("identify_start", table=str(delta_table))

    if dry_run:
        log.info("identify_dry_run", msg="Config valid, would identify flights")
        return

    df = read_delta_table(delta_table)
    df = _assign_flight_ids(df, gap_threshold_s)
    df = _join_flightlist(df, cfg.paths.resolve("download_dir"))
    df = df.drop(["_callsign", "_dt_s", "_seg_global", "_seg_idx"])

    write_columns(df, delta_table)

    log.info(
        "identify_done",
        flights=df["meta_original_flight_id"].n_unique(),
        segments=df["meta_flight_id"].n_unique(),
        rows=len(df),
    )


# ---------------------------------------------------------------------------
# Command 3 — preprocess  (script 03)
# ---------------------------------------------------------------------------


class _ExtendedDecoder:
    """Decode EHS data and expand BDS columns."""

    def __init__(self, rawdata: object = None) -> None:
        self.rawdata = rawdata

    def __call__(self, flight: Flight) -> Flight | None:
        from datetime import timedelta

        import pandas as pd
        from traffic.core import Flight as _Flight

        if flight.duration < timedelta(minutes=4):
            return None
        decoded = flight.query_ehs(self.rawdata)
        for bds in ("bds40", "bds50", "bds60"):
            if bds not in decoded.data.columns:
                return None
        exp60 = decoded.data["bds60"].apply(pd.Series)
        exp50 = (
            decoded.data["bds50"]
            .apply(pd.Series)
            .drop(
                columns=["groundspeed", "track"],
            )
        )
        exp40 = decoded.data["bds40"].apply(pd.Series)
        result = pd.concat(
            [
                decoded.data.drop(columns=["bds40", "bds50", "bds60"]),
                exp40,
                exp50,
                exp60,
            ],
            axis=1,
        )
        drop_cols = [
            "metadata",
            "squawk",
            "bds20",
            "bds17",
            "bds18",
            "bds19",
            "bds21",
            "bds45",
            "bds10",
            "bds44",
            "bds30",
            0,
            "bds",
            "serials",
            "alert",
            "spi",
            "geoaltitude",
            "vrate_barometric",
            "vrate_inertial",
            "barometric_setting",
            "selected_fms",
            "target_source",
            "df",
            "frame",
            "onground",
        ]
        return _Flight(result.drop(columns=drop_cols, errors="ignore"))


class _RawEHSDecoder:
    """Decode EHS data for download step — no preprocessing filters.

    Unlike :class:`_ExtendedDecoder`, this decoder does **not** filter by
    flight duration, preserves ``selected_fms`` (needed for
    ``bds_fms_sel_alt_ft``), and returns the raw flight on decode failure
    instead of ``None``.
    """

    def __init__(self, rawdata: object = None) -> None:
        self.rawdata = rawdata

    def __call__(self, flight: Flight) -> Flight | None:
        import pandas as pd
        from traffic.core import Flight as _Flight

        try:
            decoded = flight.query_ehs(self.rawdata)
        except Exception:  # noqa: BLE001
            return flight

        for bds in ("bds40", "bds50", "bds60"):
            if bds not in decoded.data.columns:
                return flight

        exp60 = decoded.data["bds60"].apply(pd.Series)
        exp50 = (
            decoded.data["bds50"]
            .apply(pd.Series)
            .drop(columns=["groundspeed", "track"], errors="ignore")
        )
        exp40 = decoded.data["bds40"].apply(pd.Series)
        result = pd.concat(
            [
                decoded.data.drop(columns=["bds40", "bds50", "bds60"]),
                exp40,
                exp50,
                exp60,
            ],
            axis=1,
        )
        drop_cols = [
            "metadata",
            "squawk",
            "bds20",
            "bds17",
            "bds18",
            "bds19",
            "bds21",
            "bds45",
            "bds10",
            "bds44",
            "bds30",
            0,
            "bds",
            "serials",
            "alert",
            "spi",
            "geoaltitude",
            "vrate_barometric",
            "vrate_inertial",
            "barometric_setting",
            "target_source",
            "df",
            "frame",
            "onground",
        ]
        return _Flight(result.drop(columns=drop_cols, errors="ignore"))


class _DistanceADEPADES:
    """Compute distance to departure/arrival airports."""

    def __init__(self, flights: pl.DataFrame) -> None:
        self._flights_pd = flights.to_pandas()

    def __call__(self, flight: Flight) -> Flight:
        from traffic.data import airports

        candidate = self._flights_pd.query(
            "icao24 == @flight.icao24 and "
            "@flight.start < lastseen and @flight.stop > firstseen and "
            "departure.notnull() and arrival.notnull()"
        )
        if candidate.shape[0] == 0:
            return flight
        adep = candidate.iloc[0].departure
        ades = candidate.iloc[0].arrival
        try:
            flight = flight.distance(airports[adep], column_name="adep_dist")
            flight = flight.distance(airports[ades], column_name="ades_dist")
        except Exception:  # noqa: BLE001
            log.debug("distance_failed", icao24=flight.icao24)
        return flight


def _split_at_gaps(
    traffic: Traffic,
    *,
    threshold: str = "30s",
    min_points: int = 40,
) -> Traffic | None:
    """Split flights at data gaps and discard short segments.

    Returns a new :class:`Traffic` of clean segments, each tagged with
    an ``original_flight_id`` column linking back to the source flight.
    Returns ``None`` if no segment survives the filter.
    """
    from traffic.core import Traffic  # lazy — optional dep

    segments: list[Flight] = []
    seg_idx = 0
    for flight in traffic:
        flight_id = f"{flight.icao24}_{flight.callsign or 'NOCALL'}"
        for seg in flight.split(threshold):
            if len(seg.data) >= min_points:
                seg = seg.assign(
                    original_flight_id=flight_id,
                    flight_id=f"{flight_id}_s{seg_idx}",
                )
                seg_idx += 1
                segments.append(seg)

    log.info(
        "gap_split_done",
        flights=len(traffic),
        segments=len(segments),
        threshold=threshold,
    )

    if not segments:
        return None
    return Traffic.from_flights(segments)


def preprocess(  # noqa: PLR0911, PLR0915
    *,
    config: Path,
    history_file: Path,
    workers: int = 1,
    dry_run: bool = False,
) -> None:
    """Preprocess a single raw ADS-B history file (EHS decode, filter, resample).

    Args:
        config: Path to the YAML config file.
        history_file: Path to a single ``history_*.parquet`` file.
        workers: Number of parallel workers for traffic processing.
        dry_run: Validate config without performing I/O.
    """
    import polars as pl

    from node_fdm_pipeline.config import PipelineConfig

    _require_traffic()

    cfg = PipelineConfig.from_yaml(config)
    preprocess_dir = cfg.paths.resolve("preprocess_dir")
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    download_dir = cfg.paths.resolve("download_dir")

    date = history_file.stem.split("_")[1]
    processed = preprocess_dir / f"processed_{date}.parquet"

    log.info("preprocess_start", history=str(history_file), date=date)

    if processed.exists():
        log.info("preprocess_skip", output=str(processed), reason="already exists")
        return

    if dry_run:
        log.info("preprocess_dry_run", msg="Config valid, would preprocess")
        return

    # Lazy imports — requires traffic
    from traffic.core import Traffic

    # --- Load data ---
    t = Traffic.from_file(history_file)
    if t is None or len(t) == 0:
        log.warning("preprocess_empty", history=str(history_file))
        return

    log.info("preprocess_loaded", flights=len(t))

    extended = download_dir / f"extended_{date}.parquet"
    flightlist = download_dir / f"flightlist_{date}.parquet"
    aircraft_csv = cfg.paths.data_dir / "aircraft_db.csv"

    ext_pl = pl.read_parquet(extended)
    fl_pl_raw = pl.read_parquet(flightlist)
    fl_pl = fl_pl_raw.filter(pl.col("departure").is_not_null() & pl.col("arrival").is_not_null())
    n_dropped = len(fl_pl_raw) - len(fl_pl)
    if n_dropped:
        log.info("flightlist_filtered", kept=len(fl_pl), dropped=n_dropped)
    aircraft_pl = pl.read_csv(aircraft_csv)

    # --- EHS decode filter (traffic interop → pandas) ---
    ext_pd = ext_pl.to_pandas()
    icao24_filtered = (
        ext_pd.groupby(["icao24"]).count().query("rawmsg > 50").reset_index().icao24.to_list()
    )
    t_ext = Traffic(t.data.query("icao24 in @icao24", local_dict={"icao24": icao24_filtered}))
    if t_ext is None:
        log.warning("preprocess_no_extended", date=date)
        return

    aircraft_pd = aircraft_pl.select("icao24", "registration", "typecode").to_pandas()
    # --- Build preprocessing pipeline (classes at module level for pickling) ---

    # --- Phase 1: Decode + Kalman filter (lazy, parallel) ---
    t_decoded = (
        t_ext.iterate_lazy(iterate_kw={"by": "1h"})
        .pipe(_ExtendedDecoder(ext_pd))
        .filter()
        .eval(desc="Phase 1 — decode + filter", max_workers=workers)
    )

    if t_decoded is None or len(t_decoded) == 0:
        log.warning("preprocess_no_flights_after_decode", date=date)
        return

    # --- Split at data gaps to avoid artificial interpolation ---
    t_segments = _split_at_gaps(t_decoded, threshold="30s", min_points=40)

    if t_segments is None:
        log.warning("preprocess_no_segments", date=date)
        return

    # --- Phase 2: Resample + aggressive filter (lazy, parallel) ---
    t_filtered = (
        t_segments.iterate_lazy()
        .resample("1s", how=None)
        .pipe(_DistanceADEPADES(fl_pl))
        .filter("aggressive")
        .resample("4s")
        .drop(
            columns=["track_unwrapped", "heading_unwrapped", "lastcontact"],
            errors="ignore",
        )
        .merge(aircraft_pd)
        .assign_id(f"{date}_{{self.typecode}}_{{idx:>05}}")
        .eval(desc="Phase 2 — resample + clean", max_workers=workers)
    )

    if t_filtered is None or len(t_filtered) == 0:
        log.warning("preprocess_no_flights_after_filter", date=date)
        return

    try:
        t_filtered = t_filtered.drop_duplicates()
    except Exception:  # noqa: BLE001
        log.debug("drop_duplicates_skipped")

    # --- Drop flights without valid ADEP/ADES distances ---
    n_before = len(t_filtered)
    data = t_filtered.data
    valid_ids = (
        data.groupby("flight_id")
        .filter(lambda g: g["adep_dist"].notna().any() and g["ades_dist"].notna().any())
        .flight_id.unique()
    )
    data = data[data.flight_id.isin(valid_ids)]
    n_after = data.flight_id.nunique()
    if n_after < n_before:
        log.info(
            "preprocess_dist_filter",
            kept=n_after,
            dropped=n_before - n_after,
        )
    if n_after == 0:
        log.warning("preprocess_no_flights_with_distances", date=date)
        return

    from traffic.core import Traffic

    t_filtered = Traffic(data)

    t_filtered.to_parquet(processed)
    log.info("preprocess_done", output=str(processed), flights=len(t_filtered))


# ---------------------------------------------------------------------------
# Command 4 — process  (script 04)
# ---------------------------------------------------------------------------


def process(  # noqa: PLR0915, PLR0912
    *,
    arch: str,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Process preprocessed flight data with full estimation pipeline.

    Pipeline stages:

    1. Load preprocessed parquet files
    2. Drop BDS / unused columns
    3. ERA5 weather interpolation (``fastmeteo``)
    4. Recompute TAS from wind + groundspeed
    5. Recompute Mach / CAS from TAS + altitude + temperature
    6. Per-flight: segment-based selected param estimation
    7. Derived columns (``fdm_gamma_rad``, ``fdm_long_wind_kt``, distance)
    8. Distance-jump cropping + adep/ades distance validation
    9. Train/val/test split by typecode

    Args:
        arch: Architecture name (opensky or qar).
        config: Path to the YAML config file.
        dry_run: Validate config without performing I/O.
    """
    import polars as pl
    from node_fdm_data.lateral import augment_lateral
    from node_fdm_data.meteo import compute_mach_and_cas, compute_tas
    from node_fdm_data.preprocessing.opensky import (
        crop_on_distance_jump,
        cumulative_distance,
        flight_processing,
    )
    from node_fdm_data.segments import build_selected_params

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)

    preprocess_dir = cfg.paths.resolve("preprocess_dir")
    process_dir = cfg.paths.resolve("process_dir")
    process_dir.mkdir(parents=True, exist_ok=True)

    log.info("process_start", arch=arch, preprocess_dir=str(preprocess_dir))

    if dry_run:
        log.info("process_dry_run", msg="Config valid, would process files")
        return

    # Find parquet files to process
    parquet_files = sorted(preprocess_dir.glob("*.parquet"))
    if not parquet_files:
        log.warning("process_empty_dir", dir=str(preprocess_dir))
        return

    # --- ERA5 weather interpolation ---
    try:
        from fastmeteo.source.arco_era5 import ArcoEra5
    except ImportError:
        log.error(
            "fastmeteo_missing",
            msg="fastmeteo is required for weather interpolation. "
            "Install with: pip install fastmeteo",
        )
        raise SystemExit(1) from None

    era5_cache = cfg.paths.resolve("era5_cache_dir")
    era5_cache.mkdir(parents=True, exist_ok=True)
    era5_features = cfg.era5_features or None
    arco_grid = ArcoEra5(local_store=str(era5_cache), features=era5_features)

    # --- Selected params config ---
    sel_config = cfg.selected_params.model_dump()

    # Columns to drop (BDS / raw unused)
    drop_cols = ["bds05", "bds18", "bds19", "bds21", "selected_fms", "target_source"]

    for file in parquet_files:
        output = process_dir / file.name
        if output.exists():
            log.info("process_skip", file=file.name, reason="already exists")
            continue

        log.info("process_file", file=file.name)
        df = pl.read_parquet(file)

        # Stage 1: Drop unused columns
        existing_drops = [c for c in drop_cols if c in df.columns]
        if existing_drops:
            df = df.drop(existing_drops)
        df = df.unique()

        # Stage 1b: Drop rows with null coordinates (AXM-511)
        n_before_coord = len(df)
        df = df.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
        n_null_coords = n_before_coord - len(df)
        if n_null_coords:
            log.info(
                "process_null_coords_dropped",
                file=file.name,
                dropped=n_null_coords,
            )

        # Stage 2: ERA5 weather interpolation (pandas interop)
        pd_df = df.to_pandas()
        if "timestamp" in pd_df.columns and hasattr(pd_df["timestamp"].dtype, "tz"):
            pd_df["timestamp"] = pd_df["timestamp"].dt.tz_localize(None)
        pd_df = arco_grid.interpolate(pd_df)
        df = pl.from_pandas(pd_df)

        # --- ERA5 validation gate (AXM-491, AXM-513) ---
        _era5_expected = cfg.era5_features or [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
        _era5_missing = [c for c in _era5_expected if c not in df.columns]
        if _era5_missing:
            log.error(
                "process_era5_validation_failed",
                file=file.name,
                missing=_era5_missing,
            )
            continue

        n_rows = len(df)
        _era5_null_counts: dict[str, int] = {}
        _era5_above_threshold: list[str] = []
        for c in _era5_expected:
            n_bad = int(df[c].is_null().sum()) + int(df[c].is_nan().sum())
            _era5_null_counts[c] = n_bad
            if n_rows > 0 and n_bad / n_rows > cfg.era5_null_threshold:
                _era5_above_threshold.append(c)

        if _era5_null_counts:
            log.info(
                "process_era5_null_counts",
                file=file.name,
                counts=_era5_null_counts,
            )

        if _era5_above_threshold:
            log.error(
                "process_era5_validation_failed",
                file=file.name,
                above_threshold=_era5_above_threshold,
                threshold=cfg.era5_null_threshold,
                counts=_era5_null_counts,
            )
            continue

        # Row-level ERA5 cleanup: drop rows with null/NaN in ERA5 columns
        n_before_era5 = len(df)
        df = df.filter(
            pl.all_horizontal(
                pl.col(c).is_not_null() & ~pl.col(c).is_nan() for c in _era5_expected
            )
        )
        n_era5_dropped = n_before_era5 - len(df)
        if n_era5_dropped:
            log.info(
                "process_era5_rows_dropped",
                file=file.name,
                dropped=n_era5_dropped,
            )

        log.info("process_era5_done", file=file.name, cols=len(df.columns))

        # Stage 3: Preserve BDS TAS/Mach before ERA5 overwrite (AXM-532)
        if "TAS" in df.columns:
            df = df.with_columns(pl.col("TAS").alias("bds_tas_kt"))
        if "Mach" in df.columns:
            df = df.with_columns(pl.col("Mach").alias("bds_mach"))

        # Stage 4: Recompute TAS from wind + GS (ERA5-derived)
        gs_col = "groundspeed" if "groundspeed" in df.columns else "raw_gs_kt"
        if "u_component_of_wind" in df.columns and "track" in df.columns:
            df = df.with_columns(
                compute_tas(gs_col, "track", "u_component_of_wind", "v_component_of_wind").alias(
                    "TAS"
                ),
            )

        # Stage 5: Recompute Mach/CAS from TAS + altitude + temperature
        tas_col = "TAS" if "TAS" in df.columns else "era_tas_kt"
        alt_col = "altitude" if "altitude" in df.columns else "raw_alt_ft"
        if tas_col in df.columns and "temperature" in df.columns:
            mach_arr, cas_arr = compute_mach_and_cas(
                df[tas_col].to_numpy(),
                df[alt_col].to_numpy(),
                df["temperature"].to_numpy(),
            )
            df = df.with_columns(
                pl.Series("Mach", mach_arr),
                pl.Series("CAS", cas_arr),
            )

        # Stage 5: Per-flight processing (segment estimation + derived columns)
        processed_flights: list[pl.DataFrame] = []
        flight_groups = df.partition_by("flight_id", maintain_order=True)

        for flight_df in flight_groups:
            try:
                # Rename + derived columns + fill nulls
                # Must run BEFORE segment estimation (fdm_gamma_rad needed)
                flight_df = flight_processing(flight_df.lazy()).collect()

                # Filter erroneous ADS-B rows that produce impossible Mach
                # flight_processing renames "Mach" → "era_mach"
                mach_col = "era_mach" if "era_mach" in flight_df.columns else "Mach"
                if mach_col in flight_df.columns:
                    flight_df = flight_df.filter(pl.col(mach_col) <= MACH_UPPER)

                # Cumulative distance
                if "latitude" in flight_df.columns and "longitude" in flight_df.columns:
                    flight_df = cumulative_distance(flight_df)

                    # Crop on distance jumps
                    flight_df = crop_on_distance_jump(flight_df)

                # Segment-based selected parameter estimation
                # (runs after flight_processing so fdm_gamma_rad, raw_vz_ftmin etc. exist)
                flight_df = build_selected_params(flight_df, sel_config)

                # Lateral dynamics augmentation — adds in_turn, track_ortho,
                # track_loxo, drift_angle, lat_wind for Neural ODE control
                flight_df = augment_lateral(flight_df)

                # Filter: only keep flights with valid adep_dist and ades_dist
                has_adep = "adep_dist" in flight_df.columns
                has_ades = "ades_dist" in flight_df.columns
                if has_adep and has_ades:
                    null_adep = flight_df["adep_dist"].is_null().sum()
                    null_ades = flight_df["ades_dist"].is_null().sum()
                    if null_adep > 0 or null_ades > 0:
                        continue  # skip flight

                if len(flight_df) > 0:
                    processed_flights.append(flight_df)

            except Exception as exc:  # noqa: BLE001
                fid = flight_df["flight_id"][0] if len(flight_df) > 0 else "unknown"
                log.warning("process_flight_error", flight_id=fid, error=str(exc))

        if not processed_flights:
            log.warning("process_no_valid_flights", file=file.name)
            continue

        result = pl.concat(processed_flights, how="diagonal_relaxed")
        result.write_parquet(output)
        log.info(
            "process_file_done",
            file=file.name,
            rows=len(result),
            flights=len(processed_flights),
            columns=len(result.columns),
        )

    # --- Create train/val/test split from data ---
    all_frames = [pl.read_parquet(f) for f in sorted(process_dir.glob("*.parquet"))]
    if not all_frames:
        log.warning("process_no_files_for_split")
        return

    combined = pl.concat(all_frames, how="diagonal_relaxed")

    # --- Save per-flight parquet files ---
    flights_dir = process_dir / "flights"
    flights_dir.mkdir(parents=True, exist_ok=True)

    import random

    rng = random.Random(42)  # noqa: S311

    split_rows: list[dict[str, str]] = []

    for tc in sorted(combined["typecode"].unique().to_list()):
        tc_flights = combined.filter(pl.col("typecode") == tc).partition_by(
            "flight_id", maintain_order=True
        )
        rng.shuffle(tc_flights)

        n = len(tc_flights)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15)) if n > 2 else 0  # noqa: PLR2004
        # rest goes to test

        for i, flight_df in enumerate(tc_flights):
            fid = flight_df["flight_id"][0]
            out_path = flights_dir / f"{fid}.parquet"
            flight_df.write_parquet(out_path)

            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            split_rows.append(
                {
                    "flight_id": fid,
                    "filepath": str(out_path.resolve()),
                    "split": split,
                    "aircraft_type": tc,
                }
            )

    split_df = pl.DataFrame(split_rows)
    split_csv = process_dir / "dataset_split.csv"
    split_df.write_csv(split_csv)
    log.info("process_split_done", flights=len(split_df), output=str(split_csv))
