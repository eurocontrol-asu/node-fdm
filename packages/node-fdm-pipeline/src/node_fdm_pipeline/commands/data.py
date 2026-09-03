"""Data pipeline commands — v3 Delta Table pipeline.

Commands: aircraft-list, download, identify, flag, enrich, derive,
segments, convert, split.  Commands requiring the ``traffic`` library
(aircraft-list, download) guard the import and print a helpful
install message if missing.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from node_fdm_pipeline.commands import _raw_cache
from node_fdm_pipeline.commands._fleet_plan import _parse_plan_day
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan, load_selection_file
from node_fdm_pipeline.commands._selection_match import MatchResult, match_selections

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from traffic.core import Flight

    from node_fdm_pipeline.config import PipelineConfig, SelectedParamConfig


def _clean_speeds_worker(
    args: tuple[str, list[str], str, dict[str, Any]],
) -> pl.DataFrame:
    """Spawn-safe worker: read a subset of flights from delta and clean them."""
    import polars as pl
    from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds

    batch_date, flight_ids, delta_path, kwargs = args
    df = pl.read_delta(delta_path).filter(
        (pl.col("meta_batch_date") == batch_date) & (pl.col("meta_flight_id").is_in(flight_ids))
    )
    clean_existing = [c for c in df.columns if c.startswith("bds_") and c.endswith("_clean")]
    if clean_existing:
        df = df.drop(clean_existing)
    parts = [
        clean_bds_speeds(f, **kwargs)
        for f in df.partition_by("meta_flight_id", maintain_order=True)
    ]
    return pl.concat(parts, how="diagonal_relaxed")


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
    "roll": "bds_roll_deg",
    "track_rate": "bds_track_rate_dps",
}

# OpenSky raw-cache columns whose dtype varies across (date, icao24) parquets
# (e.g. ``serials`` is sometimes ``List[Int64]`` and sometimes ``String``).
# Dropped on read before ``vertical_relaxed`` concat — they are unused
# downstream (``_RawEHSDecoder`` drops them anyway).
_UNSTABLE_RAW_COLUMNS: tuple[str, ...] = ("serials",)


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

    # Sample up to N distinct aircraft (icao24) per typecode
    sampled = (
        ext.filter(pl.col("typecode").is_in(cfg.typecodes))
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


def _attach_typecode(df: pl.DataFrame, aircraft_db: pl.DataFrame) -> pl.DataFrame:
    """Fill missing meta_aircraft_type from aircraft_db via icao24 lookup."""
    import polars as pl

    db_typecode = aircraft_db.select("icao24", "typecode").rename({"typecode": "_db_typecode"})
    df = df.join(db_typecode, left_on="raw_icao24", right_on="icao24", how="left")
    return df.with_columns(
        pl.coalesce("_db_typecode", "meta_aircraft_type").alias("meta_aircraft_type"),
    ).drop("_db_typecode")


opensky: Any = None


def _get_opensky() -> Any:
    """Resolve the ``traffic.data.opensky`` indirection.

    Tests can monkeypatch ``node_fdm_pipeline.commands.data.opensky`` to
    inject a fake; production code lazily imports the real one on first use.
    """
    global opensky
    if opensky is None:
        from traffic.data import opensky as _real_opensky

        opensky = _real_opensky
    return opensky


def _fetch_and_cache_window(  # noqa: PLR0913
    cfg: PipelineConfig,
    date_str: str,
    icao24_misses: list[str],
    kind: _raw_cache.Kind,
    *,
    start: datetime,
    end: datetime,
) -> None:
    """Fetch one OpenSky kind for the miss list and write each result atomically."""
    if not icao24_misses:
        return
    api = opensky if opensky is not None else _get_opensky()

    if kind == "flightlist":
        result = api.flightlist(start, end, icao24=icao24_misses, cached=False)
        if result is None:
            log.warning("download_empty", kind=kind, date=date_str)
            return
        df = _to_polars(result)
        _raw_cache.write_atomic(_raw_cache.cache_path(cfg, kind, date_str, "_"), df)
        return

    fetcher = api.history if kind == "history" else api.extended
    result = fetcher(start, end, icao24=icao24_misses, cached=False)
    if result is None:
        log.warning("download_empty", kind=kind, date=date_str)
        return

    pdf = result.data if hasattr(result, "data") else result
    for icao24 in icao24_misses:
        sub = pdf[pdf["icao24"] == icao24] if "icao24" in pdf.columns else pdf
        df = _to_polars(sub)
        _raw_cache.write_atomic(_raw_cache.cache_path(cfg, kind, date_str, icao24), df)


def _to_polars(obj: Any) -> pl.DataFrame:
    import polars as pl

    pdf = obj.data if hasattr(obj, "data") else obj
    return pl.from_pandas(pdf)


def _ensure_window_cached(
    cfg: PipelineConfig,
    date_str: str,
    icao24_list: list[str],
    *,
    force: bool = False,
) -> None:
    """Ensure (date, icao24) cache entries exist for every kind; fetch only the misses."""
    start = datetime.strptime(date_str, "%Y%m%d")
    end = start + timedelta(hours=24)

    kinds: tuple[_raw_cache.Kind, ...] = ("history", "extended", "flightlist")
    for kind in kinds:
        if force:
            misses = list(icao24_list)
        else:
            misses = [
                icao24
                for icao24 in _raw_cache.cache_misses(cfg, kind, date_str, icao24_list)
                if not _raw_cache.cache_path(cfg, kind, date_str, icao24).is_file()
            ]
        if not misses:
            continue
        _fetch_and_cache_window(cfg, date_str, misses, kind, start=start, end=end)


def _read_icao24_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip().lower() for line in Path(path).read_text().splitlines() if line.strip()}


def _read_flight_plan(path: Path) -> dict[str, list[str]]:
    """Read a flight selection into a ``{YYYYMMDD: [icao24, ...]}`` download plan.

    The default download mode takes a date range and one aircraft list, and
    fetches every aircraft on every day. That is right when you want a whole
    fleet over a period, and wrong when a selection step has already chosen
    which flights matter: on a stratified seven-year selection it downloads
    essentially every day for every aircraft, then throws away most of it.
    Worse, the stratification becomes decorative — the effort spent balancing
    durations, hours and regions buys nothing if everything is fetched anyway.

    A plan makes the download proportional to what is kept. Each day fetches
    only the aircraft that actually have a selected flight that day.

    The CSV needs an ``icao24`` column and one of ``day`` or ``firstseen``;
    anything else (callsign, split, duration) is ignored here. Cache
    granularity is (date, icao24) rather than the individual flight, so two
    selected flights by the same aircraft on the same day cost one fetch.
    """
    import polars as pl

    # `infer_schema=False` reads every column as text, then only the date column
    # is cast below. A plan is consumed for its `icao24` and its date and nothing
    # else, so inferring types for the remaining columns is risk without benefit:
    # an A220 selection carries serial numbers that are numeric for most rows and
    # `unknown-c05ec7` for airframes whose register publishes no serial, and
    # inference picked int64 from the head of the file then failed on the first
    # such row — rejecting a valid plan over a column it never reads.
    frame = pl.read_csv(path, infer_schema=False)
    if "day" in frame.columns:
        day = pl.col("day")
    elif "firstseen" in frame.columns:
        day = pl.col("firstseen")
    else:
        raise SystemExit(f"{path}: needs a 'day' or 'firstseen' column to build a plan")

    grouped = (
        frame.with_columns(_parse_plan_day(day).dt.strftime("%Y%m%d").alias("_day"))
        .group_by("_day")
        .agg(pl.col("icao24").unique().alias("_icao24"))
        .sort("_day")
    )
    return {row["_day"]: row["_icao24"] for row in grouped.iter_rows(named=True)}


def _decode_one_window(
    cfg: PipelineConfig,
    date_str: str,
    icao24_demanded: list[str],
    aircraft_db: pl.DataFrame,
) -> tuple[pl.DataFrame | None, int]:
    """Decode a single (date, icao24-set) window from the raw cache.

    Returns the per-day frame (or ``None`` if nothing cached) and the
    skip count for ``(date, icao24)`` pairs absent from the cache.
    """
    import polars as pl
    from traffic.core import Flight, Traffic

    cached = [
        icao24
        for icao24 in icao24_demanded
        if _raw_cache.cache_path(cfg, "history", date_str, icao24).is_file()
    ]
    skipped = len(icao24_demanded) - len(cached)

    history_pl = _raw_cache.read_partition(
        cfg, "history", date_str, icao24_demanded, drop_columns=_UNSTABLE_RAW_COLUMNS
    )
    if history_pl.is_empty():
        return None, skipped

    extended_pl = _raw_cache.read_partition(
        cfg, "extended", date_str, cached, drop_columns=_UNSTABLE_RAW_COLUMNS
    )
    extended_pdf = extended_pl.to_pandas() if not extended_pl.is_empty() else None

    flightlist_path = _raw_cache.cache_path(cfg, "flightlist", date_str, "_")
    flightlist_pl = _raw_cache.read_parquet(flightlist_path) if flightlist_path.exists() else None

    history_pdf = history_pl.to_pandas()
    decoder = _RawEHSDecoder(extended_pdf)
    decoded_flights: list[Flight] = []
    if "icao24" in history_pdf.columns:
        # Group on (icao24, callsign), not on icao24 alone. `query_ehs` refuses
        # a Flight carrying several callsigns — "Several callsigns for this
        # flight" — and one aircraft flies several rotations a day, so grouping
        # by aircraft hands it a whole day and it raises every time. The raised
        # error was then swallowed and the flight came back with empty BDS
        # fields: on a one-day sample, 1.1M raw Comm-B messages decoded into
        # 4,311 usable rows, 0.4%, and the only two aircraft that worked were
        # the two that happened to fly a single rotation.
        #
        # (icao24, callsign) is the same key `identify` uses to build
        # meta_original_flight_id, so this anticipates its coarse split rather
        # than inventing one. identify additionally cuts on time gaps, which
        # query_ehs does not care about.
        group_keys = ["icao24"]
        if "callsign" in history_pdf.columns:
            group_keys.append("callsign")
        ehs_failures = 0
        for _key, group in history_pdf.groupby(group_keys, dropna=False):
            try:
                fl = Flight(group)
            except Exception:  # noqa: BLE001
                ehs_failures += 1
                continue
            decoded = decoder(fl)
            if decoded is not None:
                decoded_flights.append(decoded)
        if ehs_failures:
            log.warning("decode_flight_construct_failed", count=ehs_failures)
    else:
        return None, skipped

    if not decoded_flights:
        return None, skipped

    merged = Traffic.from_flights(decoded_flights)
    if merged is None:
        return None, skipped
    df = pl.from_pandas(merged.data)
    df = normalize_schema(df, batch_date=date_str)
    df = join_flightlist_inline(df, flightlist_pl)
    df = _attach_typecode(df, aircraft_db)
    return df, skipped


def decode(
    *,
    config: Path,
    start_date: str,
    end_date: str,
    icao24_filter: Path | None = None,
    dry_run: bool = False,
    **_kwargs: Any,
) -> None:
    """Decode raw parquet cache into the ``flights.delta`` Delta Table.

    Reads ``data/raw/history/`` + ``data/raw/extended/`` + ``data/raw/flightlist/``
    for the demanded set (``aircraft_db.csv`` ∩ optional ``--icao24-filter``)
    and rebuilds the Delta partitioned by ``meta_batch_date``. Makes zero
    network calls — ``traffic.data.opensky`` is never imported.
    """
    import polars as pl
    from node_fdm_data.delta import write_columns

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    aircraft_db, csv_icao24 = _load_aircraft_db(cfg)

    filter_set = _read_icao24_filter(icao24_filter)
    demanded = (
        [i for i in csv_icao24 if i.lower() in filter_set] if filter_set else list(csv_icao24)
    )

    log.info(
        "decode_start",
        start_date=start_date,
        end_date=end_date,
        n_icao24=len(demanded),
    )

    if dry_run:
        log.info("decode_dry_run", msg="Config valid, would rebuild Delta")
        return

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    step = timedelta(hours=24)

    frames: list[pl.DataFrame] = []
    skipped_total = 0
    current = start
    while current < end:
        date_str = current.strftime("%Y%m%d")
        df, skipped = _decode_one_window(cfg, date_str, demanded, aircraft_db)
        skipped_total += skipped
        if df is not None:
            frames.append(df)
        current += step

    if skipped_total > 0:
        log.warning("decode_skipped_uncached", count=skipped_total)

    if not frames:
        log.info("decode_done", rows=0, msg="no cached data in range")
        return

    out = pl.concat(frames, how="diagonal_relaxed")
    delta_path = Path(cfg.paths.data_dir) / "flights.delta"
    write_columns(out, delta_path)
    log.info("decode_done", rows=len(out), output=str(delta_path))


def download(  # noqa: PLR0913
    *,
    config: Path,
    start_date: str = "",
    end_date: str = "",
    step_hours: int = 24,
    dry_run: bool = False,
    no_decode: bool = False,
    force_refresh: bool = False,
    flight_plan: Path | None = None,
) -> None:
    """Download ADS-B history and EHS data into the raw parquet cache.

    Per ``(date, icao24)``, only missing cache entries are fetched from
    OpenSky. Unless ``no_decode`` is set, ``decode`` is auto-chained
    after the cache write to produce the Delta Table.

    Two modes, and the second is what a selection step wants.

    **Range mode** (``start_date``/``end_date``) fetches every aircraft in
    ``aircraft_db.csv`` on every day of the range. Right for taking a whole
    fleet over a period.

    **Plan mode** (``flight_plan``) reads a selection CSV and fetches, for each
    day, only the aircraft that have a selected flight that day. The download
    then costs what the selection keeps rather than what the range spans — on a
    stratified multi-year selection the difference is one or two orders of
    magnitude, and without it the stratification is decorative, since everything
    gets fetched regardless.

    Args:
        config: Path to the YAML config file.
        start_date: Start date (YYYY-MM-DD). Range mode.
        end_date: End date (YYYY-MM-DD). Range mode.
        step_hours: Hours between download windows. Range mode.
        dry_run: Validate config without performing I/O.
        no_decode: Skip the auto-chained ``decode`` step.
        force_refresh: Bypass cache and re-fetch every (date, icao24).
        flight_plan: CSV of selected flights (``icao24`` + ``day``/
            ``firstseen``). Plan mode; mutually exclusive with a date range.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)

    if flight_plan is not None:
        if start_date or end_date:
            raise SystemExit("--flight-plan and --start-date/--end-date are exclusive")
        plan = _read_flight_plan(flight_plan)
        if not plan:
            raise SystemExit(f"{flight_plan}: no flights to download")
        days = sorted(plan)
        # decode() still works on a range, so derive one covering the plan.
        start_date = f"{days[0][:4]}-{days[0][4:6]}-{days[0][6:]}"
        last = datetime.strptime(days[-1], "%Y%m%d") + timedelta(days=1)
        end_date = last.strftime("%Y-%m-%d")
        fetches = sum(len(v) for v in plan.values())
        log.info(
            "download_plan",
            days=len(plan),
            aircraft_days=fetches,
            first=days[0],
            last=days[-1],
        )
    else:
        if not (start_date and end_date):
            raise SystemExit("give either --flight-plan or --start-date and --end-date")
        _aircraft_db, icao24_list = _load_aircraft_db(cfg)
        plan = None
        log.info("download_start", start_date=start_date, end_date=end_date)

    if dry_run:
        log.info("download_dry_run", msg="Config valid, would download to raw cache")
        return

    _require_traffic()

    if plan is not None:
        for date_str, aircraft in sorted(plan.items()):
            log.info("download_fetch", date=date_str, aircraft=len(aircraft))
            _ensure_window_cached(cfg, date_str, aircraft, force=force_refresh)
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        step = timedelta(hours=step_hours)

        current = start
        while current < end:
            date_str = current.strftime("%Y%m%d")
            log.info("download_fetch", date=date_str)
            _ensure_window_cached(cfg, date_str, icao24_list, force=force_refresh)
            current += step

    if no_decode:
        log.info("download_done_no_decode")
        return

    decode(config=config, start_date=start_date, end_date=end_date)


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


def _load_identify_selection(config: Path, selection: Path | None) -> SelectionPlan | None:
    if selection is not None:
        return load_selection_file(selection).plan

    config_path = Path(config)
    variant = config_path.stem.removeprefix("config").lstrip(".")
    campaign_name = f"{config_path.parent.name}__{variant}" if variant else config_path.parent.name
    default_path = config_path.parent / "results" / f"selection_{campaign_name}.csv"
    if not default_path.exists():
        return None
    return load_selection_file(default_path).plan


def _matchable_rotations(df: pl.DataFrame) -> tuple[dict[str, object], ...]:
    import polars as pl

    rotations = df
    if "icao24" not in rotations.columns and "raw_icao24" in rotations.columns:
        rotations = rotations.with_columns(pl.col("raw_icao24").alias("icao24"))
    if "callsign" not in rotations.columns and "raw_callsign" in rotations.columns:
        rotations = rotations.with_columns(pl.col("raw_callsign").alias("callsign"))

    rows = rotations.to_dicts()
    for row in rows:
        for field in ("firstseen", "lastseen"):
            value = row.get(field)
            if value is None:
                value = row.get("raw_timestamp")
            if isinstance(value, datetime):
                row[field] = int(value.timestamp())
            elif isinstance(value, float) and value.is_integer():
                row[field] = int(value)
            elif isinstance(value, str):
                try:
                    row[field] = int(value)
                except ValueError:
                    row[field] = int(
                        datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                    )
    return tuple(rows)


def identify(
    *,
    config: Path,
    selection: Path | None = None,
    gap_threshold_s: int = 30,
    dry_run: bool = False,
) -> MatchResult | None:
    """Identify flights: segment at gaps and assign flight IDs.

    Reads the Delta Table produced by ``download`` (which already contains
    flightlist metadata), detects temporal gaps within each
    (icao24, callsign) group, and assigns ``meta_flight_id`` with segment
    suffixes.

    Short segments are **not** filtered — they are flagged at étape 2.

    Args:
        config: Path to the YAML config file.
        selection: Optional recorded campaign selection (JSON rows or CSV).
        gap_threshold_s: Gap threshold in seconds for segment splitting.
        dry_run: Validate config without modifying the Delta Table.
    """
    import polars as pl
    from node_fdm_data.delta import read_delta_table, write_columns

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("identify_start", table=str(delta_table))

    if dry_run:
        log.info("identify_dry_run", msg="Config valid, would identify flights")
        return None

    df = read_delta_table(delta_table)
    campaign_days: tuple[str, ...] = (
        tuple(str(day) for day in df["meta_batch_date"].unique().to_list())
        if "meta_batch_date" in df.columns
        else ()
    )
    selection_plan: SelectionPlan | None = None
    match_result: MatchResult | None = None
    selection_plan = _load_identify_selection(config, selection)

    if selection_plan is not None:
        match_result = match_selections(selection_plan, _matchable_rotations(df))
        if match_result.admitted:
            accepted = tuple(
                flight
                for flight in selection_plan.flights
                if flight.selection_id not in match_result.rejections
            )
            identity = pl.DataFrame(
                {
                    "selection_id": [flight.selection_id for flight in accepted],
                    "icao24": [flight.icao24 for flight in accepted],
                    "callsign": [flight.callsign for flight in accepted],
                    "firstseen": [flight.firstseen for flight in accepted],
                    "lastseen": [flight.lastseen for flight in accepted],
                    "msn": [flight.msn for flight in accepted],
                    "split": [flight.split for flight in accepted],
                    "cohorts": [tuple(sorted(flight.cohorts)) for flight in accepted],
                    "utc_days": [flight.utc_days for flight in accepted],
                }
            )
            admitted = pl.DataFrame(match_result.admitted)
            superseded = [column for column in identity.columns if column in admitted.columns]
            df = pl.concat([admitted.drop(superseded), identity], how="horizontal")
        else:
            df = df.head(0)

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

    if selection_plan is not None and match_result is not None:
        absent_ids = {
            rejection.selection_id
            for rejection in match_result.rejections.values()
            if rejection.kind == "absent"
        }
        absence_root = _raw_cache.cache_root(cfg, "history")
        for day in campaign_days:
            expected_absences = [
                flight.selection_id
                for flight in selection_plan.flights
                if flight.selection_id in absent_ids and day in flight.utc_days
            ]
            if expected_absences:
                _raw_cache.publish_absence(absence_root, day, "history", expected_absences)

    return match_result


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

    identity_columns = [
        column
        for column in (
            "selection_id",
            "cohort",
            "meta_selection_day",
            "meta_source_day",
            "msn",
            "meta_msn",
            "split",
            "meta_split",
        )
        if column in df.columns
    ]
    identity = df.select("meta_flight_id", *identity_columns).unique(subset=["meta_flight_id"])

    result = preprocess_flights(
        df,
        rate_s=cfg.preprocess.rate_s,
        max_gap_s=cfg.preprocess.max_gap_s,
        min_duration_s=cfg.preprocess.min_duration_s,
        smooth=cfg.preprocess.smooth,
    )
    missing_identity = [column for column in identity_columns if column not in result.columns]
    if missing_identity:
        result = result.join(
            identity.select("meta_flight_id", *missing_identity),
            on="meta_flight_id",
            how="left",
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


def _slice_by_date(df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
    # Dates are YYYY-MM-DD; end_date is exclusive, matching download and decode.
    # An empty bound leaves that side open.
    import polars as pl

    day = pl.col("raw_timestamp").dt.date()
    out = df
    if start_date:
        out = out.filter(day >= datetime.strptime(start_date, "%Y-%m-%d").date())
    if end_date:
        out = out.filter(day < datetime.strptime(end_date, "%Y-%m-%d").date())
    return out


@dataclass(frozen=True)
class EnrichOutcome:
    """Result of one bounded ERA5 enrichment write."""

    rows: int
    era_columns: tuple[str, ...]
    max_null_fraction: float


ENRICH_INPUT_COLUMNS = (
    "raw_timestamp",
    "raw_lat_deg",
    "raw_lon_deg",
    "raw_alt_ft",
    "raw_gs_kt",
    "raw_track_deg",
)
_ERA_SOURCE_COLUMNS = ("era_temp_K", "era_u_wind_ms", "era_v_wind_ms")
_ERA_DERIVED_COLUMNS = ("era_tas_kt", "era_mach", "era_cas_kt")


def _read_enrichment_window(delta_table: Path, start_date: str, end_date: str) -> pl.DataFrame:
    """Read only the requested Delta partitions instead of the whole cohort."""
    import polars as pl
    from node_fdm_data.delta import read_delta_table

    if not start_date and not end_date:
        return read_delta_table(delta_table)

    scan = pl.scan_delta(str(delta_table))
    batch_day = pl.col("meta_batch_date")
    if start_date:
        scan = scan.filter(batch_day >= start_date.replace("-", ""))
    if end_date:
        scan = scan.filter(batch_day < end_date.replace("-", ""))
    return scan.collect()


def _eligible_null_fractions(
    df: pl.DataFrame,
    outputs: tuple[str, ...],
    inputs: tuple[str, ...],
) -> dict[str, float]:
    import polars as pl

    eligible = df.filter(pl.all_horizontal(pl.col(column).is_not_null() for column in inputs))
    if eligible.is_empty():
        return dict.fromkeys(outputs, 0.0)
    values = eligible.select(
        [
            (pl.col(column).is_null() | pl.col(column).is_nan()).mean().alias(column)
            for column in outputs
        ]
    ).row(0, named=True)
    return {column: float(value or 0.0) for column, value in values.items()}


def validate_enriched_frame(df: pl.DataFrame, null_threshold: float) -> EnrichOutcome:
    """Validate ERA5 output among rows whose required ADS-B inputs exist."""

    era_columns = tuple(c for c in df.columns if c.startswith("era_"))
    required = {*ENRICH_INPUT_COLUMNS, *_ERA_SOURCE_COLUMNS, *_ERA_DERIVED_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"ERA5 validation missing columns: {', '.join(missing)}")
    if df.is_empty():
        return EnrichOutcome(rows=0, era_columns=era_columns, max_null_fraction=0.0)

    source_inputs = ENRICH_INPUT_COLUMNS[:4]
    derived_inputs = (*source_inputs, *ENRICH_INPUT_COLUMNS[4:], *_ERA_SOURCE_COLUMNS)
    groups = (
        (_ERA_SOURCE_COLUMNS, source_inputs),
        (_ERA_DERIVED_COLUMNS, derived_inputs),
    )
    fractions: dict[str, float] = {}
    for outputs, inputs in groups:
        fractions.update(_eligible_null_fractions(df, outputs, inputs))

    worst_column, worst_fraction = max(fractions.items(), key=lambda item: item[1])
    if worst_fraction > null_threshold:
        raise RuntimeError(
            f"ERA5 null fraction {worst_fraction:.3%} in {worst_column} exceeds "
            f"configured threshold {null_threshold:.3%}"
        )
    return EnrichOutcome(
        rows=len(df),
        era_columns=era_columns,
        max_null_fraction=worst_fraction,
    )


def enrich_with_grid(
    cfg: PipelineConfig,
    arco_grid: Any,
    *,
    start_date: str = "",
    end_date: str = "",
) -> EnrichOutcome:
    """Enrich one bounded cohort window with an already-shared ERA5 grid."""
    from node_fdm_data.delta import write_columns
    from node_fdm_data.meteo import enrich_era5

    delta_table = cfg.paths.resolve("delta_table")
    df = _read_enrichment_window(delta_table, start_date, end_date)
    if df.is_empty():
        return EnrichOutcome(rows=0, era_columns=(), max_null_fraction=0.0)

    era_existing = [c for c in df.columns if c.startswith("era_")]
    if era_existing:
        log.info("enrich_drop_existing", columns=era_existing)
        df = df.drop(era_existing)

    enriched = enrich_era5(df, arco_grid)
    outcome = validate_enriched_frame(enriched, cfg.era5_null_threshold)
    write_columns(enriched, delta_table)
    return outcome


def enrich(
    *,
    config: Path,
    start_date: str = "",
    end_date: str = "",
    dry_run: bool = False,
) -> None:
    """Enrich the Delta Table with ERA5 weather data (étape 3).

    Interpolates ERA5 temperature and wind components via ``fastmeteo``,
    then computes ``era_tas_kt``, ``era_mach``, and ``era_cas_kt``.
    All rows are enriched — including those flagged invalid — because
    ERA5 API calls are expensive and should not be repeated.

    Existing ``bds_*`` columns are never modified.

    **Enrich one date at a time on a multi-date table.** ``fastmeteo`` fills its
    local store hour by hour from the earliest to the latest timestamp it is
    handed (``Grid.sync_local``), with no notion of which days are actually
    wanted. Passing a table spanning February to November therefore asks it to
    download every hour of those 258 days -- thousands of global hourly fields,
    at roughly 4 GB per day -- rather than the four days the table holds. That
    fills a disk long before it finishes.

    Bounding the window makes the download proportional to the data: one day in,
    one day fetched. ``write_columns`` merges rather than replaces, so enriching
    day by day accumulates correctly, and the local ERA5 store can be pruned
    between days since the interpolated values live in the table afterwards.

    Args:
        config: Path to the YAML config file.
        start_date: Restrict to rows on or after this date (YYYY-MM-DD).
        end_date: Restrict to rows strictly before this date (YYYY-MM-DD).
        dry_run: Validate config without modifying the Delta Table.
    """
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

    outcome = enrich_with_grid(
        cfg,
        arco_grid,
        start_date=start_date,
        end_date=end_date,
    )
    if outcome.rows == 0:
        log.warning("enrich_empty_window", start_date=start_date, end_date=end_date)
        return
    if start_date or end_date:
        log.info("enrich_window", start_date=start_date, end_date=end_date, rows=outcome.rows)
    log.info(
        "enrich_done",
        rows=outcome.rows,
        era_cols=outcome.era_columns,
        max_null_fraction=outcome.max_null_fraction,
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

    if not delta_table.exists():
        # Fallback: typecode-partitioned layout (preprocess_dir/<typecode> ->
        # process_dir/<typecode>).  Used by light pipelines and the e2e
        # config-threading harness.
        _derive_typecode_partitioned(cfg)
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

    df = derive_columns(
        df,
        airport_coords=airport_coords,
        lateral_cfg=cfg.lateral_detection.to_params(),
    )

    write_columns(df, delta_table)

    derived_cols = [
        c for c in df.columns if c.startswith("fdm_") and not c.startswith("fdm_flag_")
    ]
    log.info(
        "derive_done",
        rows=len(df),
        derived_cols=derived_cols,
    )


def label_modes(
    *,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Attach the per-sample mode label column (étape 5).

    Reads the Delta Table, applies
    :func:`node_fdm_data.preprocessing.label_modes.label_modes`, and writes
    the resulting ``fdm_mode_label`` column back via
    :func:`node_fdm_data.delta.write_columns`.  Re-running drops the
    existing column first (idempotent).

    Args:
        config: Path to the YAML config file.
        dry_run: Validate config without modifying the Delta Table.
    """
    from node_fdm_data.delta import read_delta_table, write_columns
    from node_fdm_data.preprocessing.label_modes import (
        MODE_LABEL_COLUMN,
    )
    from node_fdm_data.preprocessing.label_modes import (
        label_modes as label_modes_fn,
    )

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")

    log.info("label_modes_start", table=str(delta_table))

    if dry_run:
        log.info(
            "label_modes_dry_run",
            msg="Config valid, would label modes",
            table=str(delta_table),
        )
        return

    df = read_delta_table(delta_table)
    if MODE_LABEL_COLUMN in df.columns:
        log.info("label_modes_drop_existing", column=MODE_LABEL_COLUMN)
        df = df.drop(MODE_LABEL_COLUMN)

    df = label_modes_fn(df)
    write_columns(df.select(["meta_flight_id", MODE_LABEL_COLUMN]), delta_table)

    log.info("label_modes_done", rows=len(df))


def _derive_typecode_partitioned(cfg: object) -> None:
    """Run derive on a typecode-partitioned Delta layout.

    Reads ``preprocess_dir/<typecode>`` for each ``cfg.typecodes`` entry,
    computes derived columns, and writes the result to
    ``process_dir/<typecode>``.
    """
    import polars as pl
    from node_fdm_data.preprocessing.derive import derive_columns

    preprocess_root = cfg.paths.resolve("preprocess_dir")  # type: ignore[attr-defined]
    process_root = cfg.paths.resolve("process_dir")  # type: ignore[attr-defined]
    process_root.mkdir(parents=True, exist_ok=True)

    for typecode in cfg.typecodes:  # type: ignore[attr-defined]
        in_path = preprocess_root / typecode
        if not in_path.exists():
            log.warning("derive_typecode_missing_input", typecode=typecode, path=str(in_path))
            continue
        df = pl.read_delta(str(in_path))
        df = derive_columns(df, lateral_cfg=cfg.lateral_detection.to_params())  # type: ignore[attr-defined]
        out_path = process_root / typecode
        df.write_delta(str(out_path), mode="overwrite")
        log.info("derive_typecode_done", typecode=typecode, rows=len(df))


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

    n_workers = cfg.computing.default_cpu_count
    delta_cols = pl.scan_delta(str(delta_table)).collect_schema().names()
    clean_existing = [c for c in delta_cols if c.startswith("bds_") and c.endswith("_clean")]
    if clean_existing:
        log.info("clean_speeds_drop_existing", columns=clean_existing)

    if "meta_batch_date" in delta_cols:
        meta = (
            pl.scan_delta(str(delta_table))
            .select(["meta_flight_id", "meta_batch_date"])
            .unique()
            .collect()
        )
        tasks: list[tuple[str, list[str], str, dict[str, Any]]] = []
        n_dates = meta["meta_batch_date"].n_unique()
        for (batch_date,), grp in meta.group_by("meta_batch_date"):
            ids = grp["meta_flight_id"].to_list()
            chunk_size = max(1, len(ids) // max(1, n_workers // n_dates))
            for i in range(0, len(ids), chunk_size):
                tasks.append((batch_date, ids[i : i + chunk_size], str(delta_table), cs_kwargs))

        log.info("clean_speeds_dispatch", tasks=len(tasks), workers=n_workers)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_clean_speeds_init_polars,
            mp_context=ctx,
        ) as ex:
            processed: list[pl.DataFrame] = list(ex.map(_clean_speeds_worker, tasks))
    else:
        from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds

        df = read_delta_table(delta_table)
        if clean_existing:
            df = df.drop(clean_existing)
        processed = [
            clean_bds_speeds(f, **cs_kwargs)
            for f in df.partition_by("meta_flight_id", maintain_order=True)
        ]

    df = pl.concat(processed, how="diagonal_relaxed")
    write_columns(df, delta_table)

    clean_cols = [c for c in df.columns if c.startswith("bds_") and c.endswith("_clean")]
    log.info(
        "clean_speeds_done",
        rows=len(df),
        flights=len(processed),
        clean_cols=clean_cols,
    )


def _build_selected_params_with_valid_filter(
    flight_df: pl.DataFrame,
    sel_config: dict[str, Any],
) -> pl.DataFrame:
    """Run :func:`build_selected_params` on rows where ``fdm_flag_valid`` is true.

    The reference detection scripts in ``data/figures/new_idea_segment``
    pre-filter on ``fdm_flag_valid`` BEFORE running detection. To match
    that behaviour without changing the Delta row count, we:

    1. Filter the per-flight DataFrame to valid rows (if the column exists).
    2. Run ``build_selected_params`` on the filtered slice.
    3. Identify the columns the detector PRODUCED (computed as the set
       difference between the output columns and the filtered-input columns),
       then scatter their values back into full-length arrays indexed by the
       original valid-row positions. Invalid rows receive NaN (numeric) or
       ``None`` (non-numeric).

    Pre-existing columns of the same name in ``flight_df`` (from a prior
    segments run) are dropped before merge so the freshly-produced versions
    take their place — matching the behaviour of the previous in-place call
    to ``build_selected_params``.

    When ``fdm_flag_valid`` is absent (e.g. unit-test fixtures) the function
    is run directly on the input — preserving backward compatibility.

    Args:
        flight_df: Single-flight DataFrame (sorted by time).
        sel_config: Selected-parameter config dict (passed through).

    Returns:
        DataFrame with the same row count as ``flight_df`` and all columns
        produced by ``build_selected_params`` merged in (NaN/null on invalid
        rows).
    """
    import numpy as np
    import polars as pl
    from node_fdm_data.segments import build_selected_params

    # Backward-compatible path: no flag column -> behave as before.
    if "fdm_flag_valid" not in flight_df.columns:
        return build_selected_params(flight_df, sel_config)

    n_full = len(flight_df)
    valid_mask = flight_df["fdm_flag_valid"].to_numpy()
    valid_idx = np.flatnonzero(valid_mask)

    # Fully-invalid flight, or filtered slice too short for the bilateral /
    # Butterworth detectors (filtfilt requires len > padlen ~ 15): skip
    # detection entirely. ``pl.concat(diagonal_relaxed)`` null-pads any
    # missing produced columns relative to other flights.
    min_valid_rows_for_detection = 32
    if valid_idx.size < min_valid_rows_for_detection:
        return flight_df

    filtered = flight_df.filter(pl.col("fdm_flag_valid"))
    filtered_input_cols = set(filtered.columns)
    produced = build_selected_params(filtered, sel_config)

    # Columns the detector added (set difference) plus any pre-existing
    # ``fdm_*_sel*`` / ``fdm_*_target*`` columns that the detector rewrites
    # in-place (same name in input and output, but the values come from the
    # detector). Treating the latter as "produced" ensures stale values from
    # a prior segments run are overwritten on the full-length frame, not
    # left in place by the merge-back step.
    detector_overwritten = {
        c
        for c in produced.columns
        if c.startswith("fdm_") and ("_sel" in c or "_target" in c) and not c.endswith("_known")
    }
    # Also include the boolean "_known" companion produced by the detector
    # (e.g. ``fdm_tas_target_known``).
    detector_overwritten |= {
        c for c in produced.columns if c.startswith("fdm_") and c.endswith("_target_known")
    }
    new_cols = [
        c for c in produced.columns if c not in filtered_input_cols or c in detector_overwritten
    ]
    if not new_cols:
        return flight_df

    expansions: list[pl.Series] = []
    for col in new_cols:
        src = produced[col]
        dtype = src.dtype
        if dtype.is_numeric():
            src_np = src.to_numpy().astype(np.float64, copy=False)
            full = np.full(n_full, np.nan, dtype=np.float64)
            full[valid_idx] = src_np
            expansions.append(pl.Series(col, full).cast(dtype))
        else:
            src_list = src.to_list()
            full_list: list[Any] = [None] * n_full
            for pos, val in zip(valid_idx, src_list, strict=True):
                full_list[int(pos)] = val
            expansions.append(pl.Series(col, full_list, dtype=dtype))

    # Drop any same-named pre-existing columns before merge so the produced
    # versions replace them cleanly.
    drop_cols = [c for c in new_cols if c in flight_df.columns]
    if drop_cols:
        flight_df = flight_df.drop(drop_cols)
    return flight_df.with_columns(expansions)


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

    # Segment detection is per-flight (row-iterative). The detector is run
    # on rows passing ``fdm_flag_valid`` only; invalid-row outputs are
    # NaN-merged back so the Delta row count is preserved. Each partition
    # is re-sorted by raw_timestamp because partition_by(..., maintain_order=True)
    # only preserves group order, not row order WITHIN a group — Delta storage
    # may interleave rows from multiple batch files. Without this sort the
    # last-point anchor in `_anchored_target` lands on the wrong row.
    flights = df.partition_by("meta_flight_id", maintain_order=True)
    processed: list[pl.DataFrame] = []
    for flight_df in flights:
        flight_df = flight_df.sort("raw_timestamp")
        flight_df = _build_selected_params_with_valid_filter(flight_df, sel_config)
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


def _segments_run(
    *,
    input_path: str,
    output_path: str,
    selected_params: SelectedParamConfig,
) -> None:
    """Run segment detection on a stand-alone Delta table I/O boundary.

    Reads the Delta table at *input_path*, partitions per ``flight_id`` (or
    ``meta_flight_id`` when present), runs :func:`build_selected_params` per
    flight with *selected_params*, and writes the result to *output_path*.
    Used by integration tests that exercise the on-disk contract (e.g.
    ``fdm_tas_target_known`` emission, no-global-backfill invariant) without
    needing the full pipeline config plumbing.

    *selected_params* is required rather than defaulted: its detector
    hyper-parameters are a calibration result, so the caller must state which
    one it is running (see the note in ``node_fdm_pipeline.config``).
    """
    import deltalake
    import polars as pl

    df = pl.read_delta(input_path)
    sel_config = selected_params.model_dump()

    partition_col: str | None = None
    for candidate in ("meta_flight_id", "flight_id"):
        if candidate in df.columns:
            partition_col = candidate
            break

    flights = (
        df.partition_by(partition_col, maintain_order=True) if partition_col is not None else [df]
    )
    processed = [
        _build_selected_params_with_valid_filter(flight, sel_config) for flight in flights
    ]
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

# rs1090 source keys that `_BDS_RENAME` maps to `bds_*` targets.  When EHS
# decoding fails for every flight in a window, downstream `normalize_schema`
# would otherwise produce zero `bds_*` columns and break the schema contract
# consumed by `clean-speeds` / `derive`.  Each source key is paired with the
# numpy dtype it carries on the success path, so null-fills survive Delta's
# unsupported-Null-dtype check.
_BDS_SOURCE_KEYS: tuple[str, ...] = tuple(_BDS_RENAME.keys())
_BDS_SOURCE_DTYPES: dict[str, str] = {
    "selected_mcp": "Float64",
    "selected_fms": "Float64",
    "IAS": "Float64",
    "TAS": "Float64",
    "Mach": "Float64",
    "heading": "Float64",
    "roll": "Float64",
    "track_rate": "Float64",
}


def _explode_bds_column(series: pd.Series, keys: tuple[str, ...]) -> pd.DataFrame:
    """Explode a Series of rs1090 dicts into a flat DataFrame.

    Drop-in replacement for ``series.apply(pd.Series)`` that avoids pandas'
    super-linear behavior on object-dtype dict columns.

    Numeric keys are cast to float explicitly rather than left to pandas'
    inference, because inference on a list of ``object`` reads the *contents*,
    not the contract: a key present on millions of rows lands as ``float64``
    while one present on a handful lands as ``object``, which Delta then stores
    as a string. That is how ``bds_fms_alt_sel_ft`` — 40 non-null values out of
    13,037,916 — reached ``segments`` holding ``'31000.0'`` instead of
    ``31000.0`` and failed it on ``is_not_nan`` over dtype ``str``, while its
    sibling ``bds_mcp_alt_sel_ft`` (3.7 M values) was fine.

    The dtype each key must carry is already declared in
    :data:`_BDS_SOURCE_DTYPES`, which the decode-failure path honours. This
    applies the same contract on the success path, so a column's type no longer
    depends on how often the aircraft happened to broadcast that register.
    """
    import pandas as pd

    n = len(series)
    out: dict[str, list[object | None]] = {k: [None] * n for k in keys}
    arr = series.to_numpy()
    for i, d in enumerate(arr):
        if isinstance(d, dict):
            for k in keys:
                out[k][i] = d.get(k)

    frame = pd.DataFrame(out, index=series.index)
    for key in keys:
        if _BDS_SOURCE_DTYPES.get(key) == "Float64":
            # errors="coerce": a malformed value becomes NaN rather than pinning
            # the whole column back to object and reintroducing the bug.
            frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame


def _flight_with_empty_bds_keys(flight: Flight) -> Flight:
    """Return *flight* with every BDS source key present (null-filled if absent).

    Guarantees the schema contract that `normalize_schema` relies on: even
    when EHS decoding fails or yields incomplete subframes, the resulting
    Flight's DataFrame carries every key in :data:`_BDS_SOURCE_KEYS`, so the
    downstream rename produces every `bds_*` column (with null values).
    """
    import pandas as pd
    from traffic.core import Flight as _Flight

    df = flight.data
    missing = [k for k in _BDS_SOURCE_KEYS if k not in df.columns]
    if not missing:
        return flight
    n = len(df)
    df = df.assign(**{k: pd.array([pd.NA] * n, dtype=_BDS_SOURCE_DTYPES[k]) for k in missing})
    return _Flight(df)


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
        except Exception as exc:  # noqa: BLE001
            # Log rather than swallow. A silent fallback here cost 99.6% of the
            # selected-parameter signal without a single line of output, and the
            # cause ("Several callsigns for this flight") was one string away
            # from being obvious.
            log.warning(
                "decode_ehs_failed",
                icao24=str(flight.icao24),
                callsign=str(flight.callsign),
                error=str(exc)[:120],
            )
            return _flight_with_empty_bds_keys(flight)

        for bds in ("bds40", "bds50", "bds60"):
            if bds not in decoded.data.columns:
                return _flight_with_empty_bds_keys(flight)

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

    **Superseded where a stratified selection exists.** The v2 fleet pipeline
    draws the split before any trajectory is fetched (``stratify.py``), keyed on
    **MSN** and balanced across year, hour, duration and region; carry it onto the
    table with ``split-from-selection`` instead. Hashing ``raw_icao24`` is unsafe
    against that selection — 103 MSNs in the v2 fleet hold more than one Mode-S
    address, so one airframe would land in two splits while an icao24-keyed leak
    check still reads zero. This command refuses to overwrite a ``meta_split``
    that is already present rather than silently replacing a stratified split
    with a hashed one.

    Args:
        config: Path to the YAML config file.
        ratios: ``(train, val, test)`` proportions.
        seed: Hash salt for reproducible splits.
        dry_run: Validate config without modifying the Delta Table.

    Raises:
        SystemExit: If the table already carries ``meta_split``.
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
    if "meta_split" in df.columns:
        raise SystemExit(
            f"{delta_table} already carries meta_split. Overwriting it with an "
            "icao24-hashed split would discard a stratified, MSN-keyed one and can "
            "put a single airframe in two splits. Drop the column first if you "
            "really mean to re-derive it."
        )
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
