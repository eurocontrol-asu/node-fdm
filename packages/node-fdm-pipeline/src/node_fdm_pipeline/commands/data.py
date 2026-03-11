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
    "preprocess",
    "process",
]

log = structlog.get_logger()


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
# Command 2 — download  (script 02)
# ---------------------------------------------------------------------------


def download(
    *,
    config: Path,
    start_date: str,
    end_date: str,
    step_hours: int = 24,
    dry_run: bool = False,
) -> None:
    """Download ADS-B history data from OpenSky by date range.

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
            f"aircraft_db.csv not found at {aircraft_csv}. " "Run 'fdm aircraft-list' first."
        )

    log.info(
        "download_start",
        start_date=start_date,
        end_date=end_date,
    )

    if dry_run:
        log.info("download_dry_run", msg="Config valid, would download data")
        return

    import polars as pl

    _require_traffic()
    from traffic.data import opensky

    download_dir = cfg.paths.resolve("download_dir")
    download_dir.mkdir(parents=True, exist_ok=True)

    icao24_list = pl.read_csv(aircraft_csv)["icao24"].to_list()

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    step = timedelta(hours=step_hours)

    current = start
    while current < end:
        date_str = current.strftime("%Y%m%d")
        next_day = current + timedelta(hours=24)

        for kind, fetcher in [
            ("history", lambda s, e: opensky.history(s, e, icao24=icao24_list)),
            ("flightlist", lambda s, e: opensky.flightlist(s, e, icao24=icao24_list)),
            ("extended", lambda s, e: opensky.extended(s, e, icao24=icao24_list)),
        ]:
            path = download_dir / f"{kind}_{date_str}.parquet"
            if path.exists():
                log.info("download_skip", kind=kind, date=date_str, reason="exists")
                continue

            log.info("download_fetch", kind=kind, date=date_str)
            result = fetcher(current, next_day)
            if result is not None:
                result.to_parquet(path)
                log.info("download_saved", kind=kind, path=str(path))
            else:
                log.warning("download_empty", kind=kind, date=date_str)

        current += step


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


def process(  # noqa: PLR0915
    *,
    arch: str,
    config: Path,
    dry_run: bool = False,
) -> None:
    """Process preprocessed flight data and create train/val/test split.

    Applies ``flight_processing()`` from the architecture's preprocessing
    module to each preprocessed parquet file, then runs ``split_by_icao()``
    to create a deterministic train/val/test partition.

    Args:
        arch: Architecture name (opensky or qar).
        config: Path to the YAML config file.
        dry_run: Validate config without performing I/O.
    """
    import polars as pl

    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(config)
    info = resolve_architecture(arch)

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

    # Process each file through architecture-specific flight_processing
    for file in parquet_files:
        output = process_dir / file.name
        if output.exists():
            log.info("process_skip", file=file.name, reason="already exists")
            continue

        log.info("process_file", file=file.name)
        df = pl.read_parquet(file).lazy()
        processed = info.preprocessing_fn(df).collect()
        processed.write_parquet(output)
        log.info("process_file_done", file=file.name, rows=len(processed))

    # Create train/val/test split from data (by typecode groups)
    all_frames = [pl.read_parquet(f) for f in sorted(process_dir.glob("*.parquet"))]
    if not all_frames:
        log.warning("process_no_files_for_split")
        return

    combined = pl.concat(all_frames)
    flights = combined.select("flight_id", "typecode").unique()

    import random

    rng = random.Random(42)  # noqa: S311
    typecodes = sorted(flights["typecode"].unique().to_list())
    rng.shuffle(typecodes)

    # Assign typecodes to splits (70/15/15)
    total = len(flights)
    train_target = int(total * 0.7)
    running = 0
    train_tc: set[str] = set()
    val_tc: set[str] = set()
    for tc in typecodes:
        n = flights.filter(pl.col("typecode") == tc).height
        if running < train_target:
            train_tc.add(tc)
            running += n
        else:
            break
    remaining = [tc for tc in typecodes if tc not in train_tc]
    mid = len(remaining) // 2
    val_tc = set(remaining[:mid]) if remaining else set()

    def _assign(tc: str) -> str:
        if tc in train_tc:
            return "train"
        if tc in val_tc:
            return "val"
        return "test"

    split_series = flights["typecode"].map_elements(_assign, return_dtype=pl.Utf8)
    split_df = flights.with_columns(split_series.alias("split"))

    split_csv = process_dir / "dataset_split.csv"
    split_df.write_csv(split_csv)
    log.info("process_split_done", flights=len(split_df), output=str(split_csv))
