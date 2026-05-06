"""Data pipeline commands — v3 Delta Table pipeline.

Commands: aircraft-list, download, identify, flag, enrich, derive,
segments, convert, split.  Commands requiring the ``traffic`` library
(aircraft-list, download) guard the import and print a helpful
install message if missing.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from traffic.core import Flight

    from node_fdm_pipeline.config import PipelineConfig


def _clean_speeds_worker(args: tuple[pl.DataFrame, dict[str, Any]]) -> pl.DataFrame:
    """ProcessPool worker: invoke ``clean_bds_speeds`` on a single flight."""
    from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds

    flight_df, kwargs = args
    return clean_bds_speeds(flight_df, **kwargs)


def _clean_speeds_init_polars() -> None:
    """ProcessPool initializer: cap Polars threads to 1 inside each worker
    to avoid oversubscription (each worker is its own process, so the
    outer Polars + 8 workers x N threads each would thrash the cores).
    """
    os.environ["POLARS_MAX_THREADS"] = "1"


__all__ = [
    "aircraft_list",
    "clean_speeds",
    "convert",
    "derive",
    "download",
    "enrich",
    "flag",
    "identify",
    "preprocess",
    "segments",
    "split",
]

log = structlog.get_logger()

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
    "selected_mcp": "bds_mcp_alt_sel_ft",
    "selected_fms": "bds_fms_alt_sel_ft",
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
    query_end_date: str | None = None,
    dry_run: bool = False,
) -> None:
    """Query OpenSky for aircraft database and save to CSV.

    Args:
        config: Path to the YAML config file.
        sample_size: Max distinct aircraft per typecode.
        query_date: Start date for flight list (YYYY-MM-DD).
        query_end_date: End date (exclusive) for flight list (YYYY-MM-DD).
            Defaults to query_date + 1 day.
        dry_run: Validate config without performing I/O.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(query_date, "%Y-%m-%d")
    if query_end_date is None:
        end_dt = start_dt + timedelta(days=1)
    else:
        end_dt = datetime.strptime(query_end_date, "%Y-%m-%d")
    if end_dt <= start_dt:
        raise SystemExit(
            f"query_end_date ({query_end_date}) must be after query_date ({query_date})"
        )

    log.info(
        "aircraft_list_start",
        typecodes=cfg.typecodes,
        query_date=query_date,
        query_end_date=end_dt.strftime("%Y-%m-%d"),
    )

    if dry_run:
        log.info("aircraft_list_dry_run", msg="Config valid, would query OpenSky")
        return

    import polars as pl

    _require_traffic()
    from traffic.data import aircraft, opensky

    opensky.trino_client.connect()
    fl_pd = opensky.flightlist(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    fl = pl.from_pandas(fl_pd)

    wanted_cols = [
        c
        for c in [
            "icao24",
            "registration",
            "typecode",
            "model",
            "manufacturername",
            "operator",
            "owner",
            "engines",
            "icaoaircrafttype",
            "built",
        ]
        if c in aircraft.data.columns
    ]
    acft_db = pl.from_pandas(aircraft.data[wanted_cols])

    # Derive age (years) from built date relative to query_date
    if "built" in acft_db.columns:
        ref_date = start_dt.date()
        acft_db = (
            acft_db.with_columns(pl.col("built").cast(pl.Utf8).str.strip_chars().alias("built"))
            .with_columns(
                pl.col("built").str.to_date(format="%Y-%m-%d", strict=False).alias("_built_date"),
            )
            .with_columns(
                ((pl.lit(ref_date) - pl.col("_built_date")).dt.total_days() / 365.25)
                .round(1)
                .alias("age_years"),
            )
            .drop("_built_date")
        )
    else:
        acft_db = acft_db.with_columns(pl.lit(None).alias("age_years"))

    # Join + extract airline code from callsign
    ext = acft_db.join(fl, on="icao24", how="inner").with_columns(
        airline=pl.col("callsign").str.slice(0, 3).str.strip_chars()
    )

    # Sample up to N distinct aircraft (icao24) per typecode, age must be known
    sampled = (
        ext.filter(pl.col("typecode").is_in(cfg.typecodes))
        .filter(pl.col("age_years").is_not_null() & (pl.col("age_years") >= 0))
        .unique(subset=["icao24"], keep="first")
        .group_by("typecode")
        .map_groups(lambda g: g.sample(n=min(sample_size, len(g)), seed=42))
    )

    out_cols = [
        c
        for c in [
            "icao24",
            "registration",
            "typecode",
            "model",
            "manufacturername",
            "operator",
            "owner",
            "engines",
            "icaoaircrafttype",
            "built",
            "age_years",
            "airline",
        ]
        if c in sampled.columns
    ]
    aircraft_db = sampled.select(out_cols)
    output = data_dir / "aircraft_db.csv"
    aircraft_db.write_csv(output)
    log.info("aircraft_list_done", rows=len(aircraft_db), output=str(output))


# ---------------------------------------------------------------------------
# Command 2 — download  (script 02 → v3 étape 0)
# ---------------------------------------------------------------------------


def normalize_schema(df: pl.DataFrame, *, batch_date: str) -> pl.DataFrame:
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


def _load_aircraft_db(cfg: PipelineConfig) -> tuple[pl.DataFrame, list[str]]:
    """Load aircraft_db.csv and return the DataFrame plus its icao24 list."""
    import polars as pl

    aircraft_csv = cfg.paths.data_dir / "aircraft_db.csv"
    if not aircraft_csv.exists():
        log.error("download_missing_aircraft_db", path=str(aircraft_csv))
        raise SystemExit(
            f"aircraft_db.csv not found at {aircraft_csv}. Run 'fdm aircraft-list' first."
        )
    aircraft_db = pl.read_csv(aircraft_csv)
    return aircraft_db, aircraft_db["icao24"].unique().to_list()


def _build_window_df(
    history: object,
    extended: object,
) -> pl.DataFrame:
    """Merge OpenSky history with decoded EHS extended data into a single Polars frame."""
    import polars as pl
    from traffic.core import Traffic

    history_df = pl.from_pandas(history.data)  # type: ignore[attr-defined]
    if extended is None:
        return history_df

    ext_pd = extended.data if hasattr(extended, "data") else extended
    decoder = _RawEHSDecoder(ext_pd)
    decoded_flights: list[Flight] = []
    for flight in history:  # type: ignore[attr-defined]
        decoded = decoder(flight)
        if decoded is not None:
            decoded_flights.append(decoded)

    if not decoded_flights:
        return history_df
    merged = Traffic.from_flights(decoded_flights)
    if merged is None:
        return history_df
    return pl.from_pandas(merged.data)


def _attach_typecode(df: pl.DataFrame, aircraft_db: pl.DataFrame) -> pl.DataFrame:
    """Fill missing meta_aircraft_type from aircraft_db via icao24 lookup."""
    import polars as pl

    db_typecode = aircraft_db.select("icao24", "typecode").rename({"typecode": "_db_typecode"})
    df = df.join(db_typecode, left_on="raw_icao24", right_on="icao24", how="left")
    return df.with_columns(
        pl.coalesce("_db_typecode", "meta_aircraft_type").alias("meta_aircraft_type"),
    ).drop("_db_typecode")


def _process_window(
    current: datetime,
    next_day: datetime,
    icao24_list: list[str],
    aircraft_db: pl.DataFrame,
) -> pl.DataFrame | None:
    """Fetch and assemble one OpenSky day window into a v3-renamed DataFrame."""
    from traffic.data import opensky

    date_str = current.strftime("%Y%m%d")
    log.info("download_fetch", date=date_str)

    history = opensky.history(current, next_day, icao24=icao24_list)
    if history is None:
        log.warning("download_empty", kind="history", date=date_str)
        return None

    extended = opensky.extended(current, next_day, icao24=icao24_list)
    df = _build_window_df(history, extended)
    df = normalize_schema(df, batch_date=date_str)

    flightlist = opensky.flightlist(current, next_day, icao24=icao24_list)
    df = join_flightlist_inline(df, flightlist)
    df = _attach_typecode(df, aircraft_db)

    log.info("download_processed", date=date_str, rows=len(df))
    return df


def download(
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
    aircraft_db, icao24_list = _load_aircraft_db(cfg)

    log.info("download_start", start_date=start_date, end_date=end_date)

    if dry_run:
        log.info("download_dry_run", msg="Config valid, would download to Delta table")
        return

    import polars as pl
    from node_fdm_data.delta import write_columns

    _require_traffic()

    delta_table = cfg.paths.resolve("delta_table")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    step = timedelta(hours=step_hours)

    all_frames: list[pl.DataFrame] = []
    current = start
    while current < end:
        next_day = current + timedelta(hours=24)
        df = _process_window(current, next_day, icao24_list, aircraft_db)
        if df is not None:
            all_frames.append(df)
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


_FL_META_RENAME = {"departure": "departure", "arrival": "arrival", "typecode": "aircraft_type"}
_FL_META_COLS = ("meta_departure", "meta_arrival", "meta_aircraft_type")


def _coerce_flightlist(flightlist: object) -> pl.DataFrame | None:
    """Coerce an OpenSky flightlist to a normalized Polars frame, or None if unusable."""
    import polars as pl

    if flightlist is None:
        return None
    fl_pd = flightlist.data if hasattr(flightlist, "data") else flightlist
    fl = fl_pd if isinstance(fl_pd, pl.DataFrame) else pl.from_pandas(fl_pd)
    if len(fl) == 0 or "icao24" not in fl.columns:
        return None
    if "callsign" in fl.columns:
        fl = fl.with_columns(
            pl.col("callsign").fill_null("NOCALL").str.strip_chars().alias("callsign"),
        )
    return fl


def _join_flight_meta(df: pl.DataFrame, fl: pl.DataFrame) -> pl.DataFrame:
    """Left-join flightlist metadata (departure/arrival/typecode) onto df by (icao24, callsign)."""
    import polars as pl

    fl_cols = [c for c in ("departure", "arrival", "typecode") if c in fl.columns]
    df = df.with_columns(
        pl.col("raw_callsign").fill_null("NOCALL").str.strip_chars().alias("_fl_callsign"),
    )
    if fl_cols:
        fl_select = fl.select(["icao24", "callsign", *fl_cols]).unique(
            subset=["icao24", "callsign"],
            keep="first",
        )
        df = df.join(
            fl_select,
            left_on=["raw_icao24", "_fl_callsign"],
            right_on=["icao24", "callsign"],
            how="left",
        )
        rename = {c: f"meta_{a}" for c, a in _FL_META_RENAME.items() if c in df.columns}
        df = df.rename(rename)
    return df.drop("_fl_callsign")


def _ensure_meta_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add any missing flightlist meta_* columns as null Utf8 placeholders."""
    import polars as pl

    missing = [c for c in _FL_META_COLS if c not in df.columns]
    if not missing:
        return df
    return df.with_columns([pl.lit(None).cast(pl.Utf8).alias(c) for c in missing])


def join_flightlist_inline(df: pl.DataFrame, flightlist: object) -> pl.DataFrame:
    """Join flightlist metadata directly onto a download batch.

    Called during ``download`` to integrate departure, arrival, and typecode
    columns into the Delta Table in a single pass — no intermediate parquet.

    Args:
        df: Batch DataFrame with ``raw_icao24``, ``raw_callsign`` columns.
        flightlist: Raw flightlist from ``opensky.flightlist()`` (Traffic or
            pandas DataFrame), or ``None`` if the API returned nothing.

    Returns:
        DataFrame with ``meta_departure``, ``meta_arrival``,
        ``meta_aircraft_type`` columns added.
    """
    fl = _coerce_flightlist(flightlist)
    if fl is not None:
        df = _join_flight_meta(df, fl)
    return _ensure_meta_columns(df)


def identify(
    *,
    config: Path,
    gap_threshold_s: int = 30,
    dry_run: bool = False,
) -> None:
    """Identify flights: segment at gaps and assign flight IDs.

    Reads the Delta Table produced by ``download`` (which already contains
    flightlist metadata), detects temporal gaps within each
    (icao24, callsign) group, and assigns ``meta_flight_id`` with segment
    suffixes.

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

    # Drop existing identify columns to allow re-identification
    id_existing = [
        c for c in df.columns if c in {"meta_original_flight_id", "meta_flight_id", "_callsign"}
    ]
    if id_existing:
        log.info("identify_drop_existing", columns=id_existing)
        df = df.drop(id_existing)

    df = _assign_flight_ids(df, gap_threshold_s)
    df = df.drop(["_callsign", "_dt_s", "_seg_global", "_seg_idx"])

    write_columns(df, delta_table)

    log.info(
        "identify_done",
        flights=df["meta_original_flight_id"].n_unique(),
        segments=df["meta_flight_id"].n_unique(),
        rows=len(df),
    )


# ---------------------------------------------------------------------------
# Command 1.5 — preprocess  (étape 1.5 — gap-aware resample)
# ---------------------------------------------------------------------------


def _ensure_identify_ran(df: pl.DataFrame) -> None:
    """Exit if meta_flight_id is missing/null, signaling identify must run before preprocess."""
    if "meta_flight_id" in df.columns and not df["meta_flight_id"].is_null().all():
        return
    log.error("preprocess_missing_identify", msg="meta_flight_id is missing or all null")
    print(  # noqa: T201
        "Error: 'identify' must be run before 'preprocess'. "
        "Run 'fdm identify --config ...' first.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _drop_pre_existing(df: pl.DataFrame) -> pl.DataFrame:
    """Drop legacy pre_gap_* columns so preprocess can rewrite them cleanly."""
    pre_existing = [c for c in df.columns if c.startswith("pre_gap_")]
    if pre_existing:
        log.info("preprocess_drop_existing", columns=pre_existing)
        df = df.drop(pre_existing)
    return df


def _cast_null_columns(result: pl.DataFrame) -> pl.DataFrame:
    """Cast all-null columns to Float64 so Delta schema-merge accepts them."""
    import polars as pl

    null_cols = [c for c in result.columns if result[c].dtype == pl.Null]
    if not null_cols:
        return result
    log.info("preprocess_cast_null_cols", columns=null_cols)
    return result.with_columns([pl.col(c).cast(pl.Float64) for c in null_cols])


def _build_preprocess_write_options(result: pl.DataFrame, delta_table: Path) -> dict[str, object]:
    """Build delta write_delta options with partitioning and predicate for incremental writes."""
    options: dict[str, object] = {"schema_mode": "merge"}
    if "meta_batch_date" not in result.columns:
        return options
    options["partition_by"] = ["meta_batch_date"]
    if delta_table.exists():
        dates = result["meta_batch_date"].unique().sort().to_list()
        quoted = ", ".join(f"'{d}'" for d in dates)
        options["predicate"] = f"meta_batch_date IN ({quoted})"
    return options


def preprocess(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Resample to regular grid with gap-aware interpolation (étape 1.5).

    Detects per-column-group sub-segments, interpolates within each,
    and leaves null values between sub-segments.  Row count changes
    (irregular → 4 s grid), so the table is overwritten entirely.

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("preprocess_start", table=str(delta_table))

    if dry_run:
        log.info("preprocess_dry_run", msg="Config valid, would preprocess")
        return

    from node_fdm_data.delta import read_delta_table
    from node_fdm_data.preprocessing.resample import preprocess_flights

    df = read_delta_table(delta_table)
    _ensure_identify_ran(df)

    rows_before = len(df)
    flights_before = df["meta_flight_id"].n_unique()

    df = _drop_pre_existing(df)

    result = preprocess_flights(
        df,
        rate_s=cfg.preprocess.rate_s,
        max_gap_s=cfg.preprocess.max_gap_s,
        min_duration_s=cfg.preprocess.min_duration_s,
        smooth=cfg.preprocess.smooth,
    )

    result = _cast_null_columns(result)

    result.write_delta(
        str(delta_table),
        mode="overwrite",
        delta_write_options=_build_preprocess_write_options(result, delta_table),
    )

    log.info(
        "preprocess_done",
        rows_before=rows_before,
        rows_after=len(result),
        flights_before=flights_before,
        flights_after=result["meta_flight_id"].n_unique(),
    )


# ---------------------------------------------------------------------------
# Command 2 — flag  (étape 2 — validity flags)
# ---------------------------------------------------------------------------


def flag(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Add validity flag columns (``fdm_flag_*``) to the Delta Table.

    Reads the Delta Table produced by ``identify``, computes boolean
    quality flags per row, and writes the flag columns back.  No rows
    are deleted — the consumer filters at read time via
    ``fdm_flag_valid``.

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_data.delta import read_delta_table, write_columns
    from node_fdm_data.preprocessing.flags import compute_flags

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("flag_start", table=str(delta_table))

    if dry_run:
        log.info("flag_dry_run", msg="Config valid, would compute flags")
        return

    df = read_delta_table(delta_table)

    # Drop existing flag columns to allow re-flagging
    flag_existing = [c for c in df.columns if c.startswith("fdm_flag_")]
    if flag_existing:
        log.info("flag_drop_existing", columns=flag_existing)
        df = df.drop(flag_existing)

    df = compute_flags(
        df,
        min_points=cfg.flag.min_points,
        min_speed_kt=cfg.flag.min_speed_kt,
        distance_low_thr=cfg.flag.distance_low_thr,
        distance_upper_thr=cfg.flag.distance_upper_thr,
    )

    write_columns(df, delta_table)

    flag_cols = [c for c in df.columns if c.startswith("fdm_flag_")]
    n_valid = df["fdm_flag_valid"].sum()
    log.info(
        "flag_done",
        rows=len(df),
        flags=len(flag_cols),
        valid_rows=n_valid,
    )


def enrich(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Enrich the Delta Table with ERA5 weather data (étape 3).

    Interpolates ERA5 temperature and wind components via ``fastmeteo``,
    then computes ``era_tas_kt``, ``era_mach``, and ``era_cas_kt``.
    All rows are enriched — including those flagged invalid — because
    ERA5 API calls are expensive and should not be repeated.

    Existing ``bds_*`` columns are never modified.

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_data.delta import read_delta_table, write_columns
    from node_fdm_data.meteo import enrich_era5

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("enrich_start", table=str(delta_table))

    if dry_run:
        log.info("enrich_dry_run", msg="Config valid, would enrich with ERA5")
        return

    try:
        from fastmeteo.source.arco_era5 import ArcoEra5
    except ImportError:
        log.error(
            "fastmeteo_missing",
            msg="fastmeteo is required for ERA5 enrichment. Install with: pip install fastmeteo",
        )
        raise SystemExit(1) from None

    era5_cache = cfg.paths.resolve("era5_cache_dir")
    era5_cache.mkdir(parents=True, exist_ok=True)
    era5_features = cfg.era5_features or None
    arco_grid = ArcoEra5(local_store=str(era5_cache), features=era5_features)

    df = read_delta_table(delta_table)

    # Drop existing ERA5 columns to allow re-enrichment
    era_existing = [c for c in df.columns if c.startswith("era_")]
    if era_existing:
        log.info("enrich_drop_existing", columns=era_existing)
        df = df.drop(era_existing)

    df = enrich_era5(df, arco_grid)

    write_columns(df, delta_table)

    era_cols = [c for c in df.columns if c.startswith("era_")]
    log.info(
        "enrich_done",
        rows=len(df),
        era_cols=era_cols,
    )


def derive(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Compute derived physics columns (étape 4).

    Reads the Delta Table produced by ``enrich``, computes flight-path
    angle, longitudinal wind, altitude difference, cumulative distance,
    and airport distances, then writes the ``fdm_*`` columns back.

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_data.delta import read_delta_table, write_columns
    from node_fdm_data.preprocessing.derive import derive_columns

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("derive_start", table=str(delta_table))

    if dry_run:
        log.info("derive_dry_run", msg="Config valid, would compute derived columns")
        return

    df = read_delta_table(delta_table)

    # Drop existing derived columns to allow re-derivation (preserve fdm_flag_*
    # and fdm_tas_from_cas_kt which is an input produced by clean-speeds)
    derive_existing = [
        c
        for c in df.columns
        if c.startswith("fdm_") and not c.startswith("fdm_flag_") and c != "fdm_tas_from_cas_kt"
    ]
    if derive_existing:
        log.info("derive_drop_existing", columns=derive_existing)
        df = df.drop(derive_existing)

    # Build airport coordinate lookup (soft dependency on traffic)
    airport_coords = _build_airport_coords(df)

    df = derive_columns(df, airport_coords=airport_coords)

    write_columns(df, delta_table)

    derived_cols = [
        c for c in df.columns if c.startswith("fdm_") and not c.startswith("fdm_flag_")
    ]
    log.info(
        "derive_done",
        rows=len(df),
        derived_cols=derived_cols,
    )


def clean_speeds(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Clean BDS speed signals per flight (Hampel + ERA fill).

    Reads the Delta Table, runs
    :func:`~node_fdm_data.preprocessing.clean_speeds.clean_bds_speeds`
    on each ``meta_flight_id`` group, and writes the new
    ``bds_*_clean`` columns back.  Raw ``bds_*`` columns are never
    modified; existing ``bds_*_clean`` columns are dropped before
    recomputation, making the stage idempotent.

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    import polars as pl
    from node_fdm_data.delta import read_delta_table, write_columns

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")
    cs_cfg = cfg.clean_speeds

    log.info("clean_speeds_start", table=str(delta_table))

    if dry_run:
        log.info("clean_speeds_dry_run", msg="Config valid, would clean BDS speeds")
        return

    df = read_delta_table(delta_table)

    clean_existing = [c for c in df.columns if c.startswith("bds_") and c.endswith("_clean")]
    if clean_existing:
        log.info("clean_speeds_drop_existing", columns=clean_existing)
        df = df.drop(clean_existing)

    flights = df.partition_by("meta_flight_id", maintain_order=True)
    cs_kwargs: dict[str, Any] = {
        "bds_window": cs_cfg.bds_window,
        "era_window": cs_cfg.era_window,
        "k": cs_cfg.k,
        "n_passes": cs_cfg.n_passes,
        "interp_max_gap": cs_cfg.interp_max_gap,
        "frozen_min_run_len_mach": cs_cfg.frozen_min_run_len_mach,
        "frozen_min_run_len_ias": cs_cfg.frozen_min_run_len_ias,
        "frozen_min_run_len_tas": cs_cfg.frozen_min_run_len_tas,
        "point_jump_max_mach": cs_cfg.point_jump_max_mach,
        "point_jump_max_kt": cs_cfg.point_jump_max_kt,
        "zigzag_jump_min_mach": cs_cfg.zigzag_jump_min_mach,
        "zigzag_jump_min_kt": cs_cfg.zigzag_jump_min_kt,
        "zigzag_half_window": cs_cfg.zigzag_half_window,
        "zigzag_density_min_bds": cs_cfg.zigzag_density_min_bds,
        "zigzag_density_min_era": cs_cfg.zigzag_density_min_era,
        "on_ground_vz_threshold": cs_cfg.on_ground_vz_threshold,
        "on_ground_alt_threshold": cs_cfg.on_ground_alt_threshold,
    }

    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_clean_speeds_init_polars,
    ) as ex:
        processed: list[pl.DataFrame] = list(
            ex.map(
                _clean_speeds_worker,
                ((f, cs_kwargs) for f in flights),
            )
        )

    df = pl.concat(processed, how="diagonal_relaxed")
    write_columns(df, delta_table)

    clean_cols = [c for c in df.columns if c.startswith("bds_") and c.endswith("_clean")]
    log.info(
        "clean_speeds_done",
        rows=len(df),
        flights=len(processed),
        clean_cols=clean_cols,
    )


def segments(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Detect constant segments and build selected-parameter columns (étape 5).

    Reads the Delta Table produced by ``derive``, runs segment detection
    per flight, and writes ``fdm_*_sel`` columns plus MCP/FMS backfill
    columns back to the table.

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    import polars as pl
    from node_fdm_data.delta import read_delta_table, write_columns
    from node_fdm_data.segments import build_selected_params

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")
    sel_config = cfg.selected_params.model_dump()

    log.info("segments_start", table=str(delta_table))

    if dry_run:
        log.info("segments_dry_run", msg="Config valid, would detect segments")
        return

    df = read_delta_table(delta_table)

    # Drop existing selected-parameter columns to allow re-segmentation.
    # Only fdm_*_sel_<unit> — preserve bds_*_sel_* inputs and fdm_*_sel_known
    # boolean flags emitted by derive (e.g. fdm_track_sel_known).
    sel_existing = [
        c for c in df.columns if c.startswith("fdm_") and "_sel_" in c and not c.endswith("_known")
    ]
    if sel_existing:
        log.info("segments_drop_existing", columns=sel_existing)
        df = df.drop(sel_existing)

    # Segment detection is per-flight (row-iterative)
    flights = df.partition_by("meta_flight_id", maintain_order=True)
    processed: list[pl.DataFrame] = []
    for flight_df in flights:
        flight_df = build_selected_params(flight_df, sel_config)
        processed.append(flight_df)

    df = pl.concat(processed, how="diagonal_relaxed")

    write_columns(df, delta_table)

    sel_cols = [c for c in df.columns if "_sel" in c]
    log.info(
        "segments_done",
        rows=len(df),
        flights=len(processed),
        sel_cols=sel_cols,
    )


def _segments_run(*, input_path: str, output_path: str) -> None:
    """Run segment detection on a stand-alone Delta table I/O boundary.

    Reads the Delta table at *input_path*, partitions per ``flight_id`` (or
    ``meta_flight_id`` when present), runs :func:`build_selected_params` per
    flight with the default selected-parameter config, and writes the result
    to *output_path*.  Used by integration tests that exercise the on-disk
    contract (e.g. ``fdm_tas_target_known`` emission, no-global-backfill
    invariant) without needing the full pipeline config plumbing.
    """
    import deltalake
    import polars as pl
    from node_fdm_data.segments import build_selected_params

    from node_fdm_pipeline.config import SelectedParamConfig

    df = pl.read_delta(input_path)
    sel_config = SelectedParamConfig().model_dump()

    partition_col: str | None = None
    for candidate in ("meta_flight_id", "flight_id"):
        if candidate in df.columns:
            partition_col = candidate
            break

    flights = (
        df.partition_by(partition_col, maintain_order=True) if partition_col is not None else [df]
    )
    processed = [build_selected_params(flight, sel_config) for flight in flights]
    out = pl.concat(processed, how="diagonal_relaxed")

    deltalake.write_deltalake(output_path, out.to_arrow(), mode="overwrite")


segments.run = _segments_run  # type: ignore[attr-defined]


def _drop_existing_convert_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop any pre-existing SI conversion / derivative output columns to allow recomputation."""
    from node_fdm_data.preprocessing.convert import SI_CONVERSIONS, SI_DERIVATIVES

    targets = {t for _, _, t in SI_CONVERSIONS} | {t for _, t in SI_DERIVATIVES}
    existing = [c for c in df.columns if c in targets]
    if not existing:
        return df
    log.info("convert_drop_existing", columns=existing)
    return df.drop(existing)


def _collect_convert_output_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Return (SI columns, fdm_d_* derivative columns) produced by the convert step."""
    si_cols = [c for c in df.columns if c.endswith(("_m", "_ms"))]
    deriv_cols = [c for c in df.columns if c.startswith("fdm_d_")]
    return si_cols, deriv_cols


def convert(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Convert to SI units and compute temporal derivatives (étapes 6-7).

    Reads the Delta Table produced by ``segments``, adds SI-unit columns
    via :func:`~node_fdm_data.preprocessing.convert.convert_si` and
    per-flight derivatives via
    :func:`~node_fdm_data.preprocessing.convert.compute_derivatives`,
    then writes the new columns back.

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_data.delta import read_delta_table, write_columns
    from node_fdm_data.preprocessing.convert import compute_derivatives, convert_si

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("convert_start", table=str(delta_table))

    if dry_run:
        log.info(
            "convert_dry_run", msg="Config valid, would convert to SI and compute derivatives"
        )
        return

    df = read_delta_table(delta_table)
    df = _drop_existing_convert_columns(df)
    df = convert_si(df)
    df = compute_derivatives(df)

    write_columns(df, delta_table)

    si_cols, deriv_cols = _collect_convert_output_columns(df)
    log.info(
        "convert_done",
        rows=len(df),
        si_cols=si_cols,
        deriv_cols=deriv_cols,
    )


def _build_airport_coords(df: pl.DataFrame) -> dict[str, tuple[float, float]] | None:
    """Build ICAO → (lat, lon) mapping from ``traffic.data.airports``.

    Returns ``None`` if the ``traffic`` library is not available or
    the required columns are missing.
    """
    if "meta_departure" not in df.columns or "meta_arrival" not in df.columns:
        return None

    try:
        from traffic.data import airports
    except ImportError:
        log.warning(
            "derive_traffic_missing",
            msg="traffic not installed — airport distance columns will be NaN",
        )
        return None

    icao_codes: set[str] = set()
    for col in ("meta_departure", "meta_arrival"):
        icao_codes.update(v for v in df[col].drop_nulls().unique().to_list() if v)

    coords: dict[str, tuple[float, float]] = {}
    for icao in icao_codes:
        try:
            ap = airports[icao]
        except ValueError:
            log.warning("derive_unknown_airport", icao=icao)
            continue
        if ap is not None:
            coords[icao] = (ap.latitude, ap.longitude)

    return coords


# Keys returned by rs1090 for each Mode-S Comm-B BDS register (verified at runtime).
# Used to explode dict-typed columns into flat columns without pandas' O(n²)
# `Series.apply(pd.Series)`. See scripts/debug/bench_ehs_explode.py.
_BDS40_KEYS: tuple[str, ...] = ("selected_mcp", "selected_fms", "barometric_setting", "bds")
_BDS50_KEYS: tuple[str, ...] = ("TAS", "roll", "track", "groundspeed", "track_rate", "bds")
_BDS60_KEYS: tuple[str, ...] = (
    "IAS",
    "Mach",
    "heading",
    "vrate_barometric",
    "vrate_inertial",
    "bds",
)


def _explode_bds_column(series: pd.Series, keys: tuple[str, ...]) -> pd.DataFrame:
    """Explode a Series of rs1090 dicts into a flat DataFrame.

    Drop-in replacement for ``series.apply(pd.Series)`` that avoids pandas'
    super-linear behavior on object-dtype dict columns.
    """
    import pandas as pd

    n = len(series)
    out: dict[str, list[object | None]] = {k: [None] * n for k in keys}
    arr = series.to_numpy()
    for i, d in enumerate(arr):
        if isinstance(d, dict):
            for k in keys:
                out[k][i] = d.get(k)
    return pd.DataFrame(out, index=series.index)


class _RawEHSDecoder:
    """Decode EHS data for download step — no preprocessing filters.

    Unlike :class:`_ExtendedDecoder`, this decoder does **not** filter by
    flight duration, preserves ``selected_fms`` (needed for
    ``bds_fms_alt_sel_ft``), and returns the raw flight on decode failure
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

        exp40 = _explode_bds_column(decoded.data["bds40"], _BDS40_KEYS)
        exp50 = _explode_bds_column(decoded.data["bds50"], _BDS50_KEYS).drop(
            columns=["groundspeed", "track"], errors="ignore"
        )
        exp60 = _explode_bds_column(decoded.data["bds60"], _BDS60_KEYS)
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


def split(
    *,
    config: Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    """Add ``meta_split`` column to the Delta Table (étape 8).

    Assigns each row to train/val/test based on a deterministic hash
    of ``raw_icao24``.  All segments of the same aircraft land in the
    same split, preventing data leakage.

    Args:
        config: Path to the YAML config file.
        ratios: ``(train, val, test)`` proportions.
        seed: Hash salt for reproducible splits.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_data.delta import read_delta_table, write_columns
    from node_fdm_data.split import split_by_icao

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("split_start", table=str(delta_table), ratios=ratios, seed=seed)

    if dry_run:
        log.info("split_dry_run", msg="Config valid, would compute split")
        return

    df = read_delta_table(delta_table)
    df = split_by_icao(df, ratios=ratios, seed=seed)

    write_columns(df, delta_table)

    counts = df.group_by("meta_split").len()
    split_map = dict(zip(counts["meta_split"].to_list(), counts["len"].to_list(), strict=True))
    log.info(
        "split_done",
        rows=len(df),
        unique_icao24=df["raw_icao24"].n_unique(),
        **split_map,
    )
