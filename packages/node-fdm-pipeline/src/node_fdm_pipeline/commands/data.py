"""Data pipeline commands — v3 Delta Table pipeline.

Commands: aircraft-list, download, identify, flag, enrich, derive,
segments, convert, split.  Commands requiring the ``traffic`` library
(aircraft-list, download) guard the import and print a helpful
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
    from traffic.core import Flight

__all__ = [
    "aircraft_list",
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


def download(  # noqa: PLR0915
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

    aircraft_db = pl.read_csv(aircraft_csv)
    icao24_list = aircraft_db["icao24"].to_list()

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
                df = pl.from_pandas(merged.data)  # type: ignore[union-attr]
            else:
                df = pl.from_pandas(history.data)
        else:
            df = pl.from_pandas(history.data)

        # Rename to v3 schema and collect
        df = _rename_to_v3(df, batch_date=date_str)

        # Join flightlist metadata directly into the batch
        flightlist = opensky.flightlist(current, next_day, icao24=icao24_list)
        df = _join_flightlist_inline(df, flightlist)

        # Fill meta_aircraft_type from aircraft_db (more reliable than flightlist)
        db_typecode = aircraft_db.select("icao24", "typecode").rename({"typecode": "_db_typecode"})
        df = df.join(db_typecode, left_on="raw_icao24", right_on="icao24", how="left")
        df = df.with_columns(
            pl.coalesce("_db_typecode", "meta_aircraft_type").alias("meta_aircraft_type"),
        ).drop("_db_typecode")

        all_frames.append(df)
        log.info("download_processed", date=date_str, rows=len(df))

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


def _join_flightlist_inline(df: pl.DataFrame, flightlist: object) -> pl.DataFrame:
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
    import polars as pl

    if flightlist is not None:
        fl_pd = flightlist.data if hasattr(flightlist, "data") else flightlist
        fl = pl.from_pandas(fl_pd) if not isinstance(fl_pd, pl.DataFrame) else fl_pd

        if len(fl) > 0 and "icao24" in fl.columns:
            if "callsign" in fl.columns:
                fl = fl.with_columns(
                    pl.col("callsign").fill_null("NOCALL").str.strip_chars().alias("callsign"),
                )
            # Normalize raw_callsign for join
            df = df.with_columns(
                pl.col("raw_callsign").fill_null("NOCALL").str.strip_chars().alias("_fl_callsign"),
            )
            fl_cols = [c for c in ("departure", "arrival", "typecode") if c in fl.columns]
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
            df = df.drop("_fl_callsign")

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

    import polars as pl
    from node_fdm_data.delta import read_delta_table
    from node_fdm_data.preprocessing.resample import preprocess_flights

    df = read_delta_table(delta_table)
    rows_before = len(df)
    flights_before = df["meta_flight_id"].n_unique()

    # Drop existing preprocess columns to allow re-run
    pre_existing = [c for c in df.columns if c.startswith("pre_gap_")]
    if pre_existing:
        log.info("preprocess_drop_existing", columns=pre_existing)
        df = df.drop(pre_existing)

    result = preprocess_flights(
        df,
        rate_s=cfg.preprocess.rate_s,
        max_gap_s=cfg.preprocess.max_gap_s,
        min_duration_s=cfg.preprocess.min_duration_s,
        smooth=cfg.preprocess.smooth,
    )

    # Cast Null-typed columns (all-null after resample) to Float64 for Delta Lake
    null_cols = [c for c in result.columns if result[c].dtype == pl.Null]
    if null_cols:
        log.info("preprocess_cast_null_cols", columns=null_cols)
        result = result.with_columns(
            [pl.col(c).cast(pl.Float64) for c in null_cols],
        )

    # Overwrite entire table (row count changes with resampling)
    delta_write_options: dict[str, object] = {"schema_mode": "merge"}
    if "meta_batch_date" in result.columns:
        delta_write_options["partition_by"] = ["meta_batch_date"]

    result.write_delta(
        str(delta_table),
        mode="overwrite",
        delta_write_options=delta_write_options,
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

    # Drop existing derived columns to allow re-derivation (preserve fdm_flag_*)
    derive_existing = [
        c for c in df.columns if c.startswith("fdm_") and not c.startswith("fdm_flag_")
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

    # Drop existing selected-parameter columns to allow re-segmentation
    # Only fdm_*_sel* — preserve bds_*_sel_* input columns
    sel_existing = [c for c in df.columns if c.startswith("fdm_") and "_sel" in c]
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

    # Drop existing SI and derivative columns to allow re-conversion
    from node_fdm_data.preprocessing.convert import SI_CONVERSIONS, SI_DERIVATIVES

    si_targets = {target for _, _, target in SI_CONVERSIONS}
    deriv_targets = {target for _, target in SI_DERIVATIVES}
    convert_existing = [c for c in df.columns if c in si_targets or c in deriv_targets]
    if convert_existing:
        log.info("convert_drop_existing", columns=convert_existing)
        df = df.drop(convert_existing)

    df = convert_si(df)
    df = compute_derivatives(df)

    write_columns(df, delta_table)

    si_cols = [c for c in df.columns if c.endswith(("_m", "_ms"))]
    deriv_cols = [c for c in df.columns if c.startswith("fdm_d_")]
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
