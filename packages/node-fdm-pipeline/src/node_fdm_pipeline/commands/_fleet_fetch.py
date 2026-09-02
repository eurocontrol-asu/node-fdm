"""Fetch one mutualised date and dispatch its rows into the owning silos.

The unit of work is a date, not a (cohort, date) pair: one request names every
aircraft any cohort wants that day, and the result is split by icao24 and written
into whichever silo owns each one. Dates are independent, so
:func:`download_fleet` runs a few of them concurrently — bounded, because the
OpenSky Trino cluster limits concurrent queries per account and exceeding it
earns rejections rather than throughput.

Two properties make the concurrency safe without any lock:

- **Disjoint writes.** Silos are strictly disjoint in icao24 (asserted in
  :func:`build_fleet_plan`), and each worker owns a distinct date, so no two
  workers ever target the same path.
- **Atomic publication.** ``_raw_cache.write_atomic`` writes a ``.tmp`` beside
  the target and ``os.rename``s it, so a reader never observes a partial file and
  an interrupted worker leaves no half-written cache entry.

Resumption is inherited unchanged: misses are computed per (silo, kind, date,
icao24), so a date whose entries all exist is skipped without a request, and a
killed run costs only the dates it had in flight.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from node_fdm_pipeline.commands import _raw_cache
from node_fdm_pipeline.commands._fleet_boundary import (
    AcquisitionPreflight,
    acquisition_section,
    preflight_acquisition,
)
from node_fdm_pipeline.commands._fleet_digest import DigestInput, ResumeDigest
from node_fdm_pipeline.config import FleetRunConfig

if TYPE_CHECKING:
    from node_fdm_pipeline.commands._fleet_plan import Cohort, FleetPlan

log = structlog.get_logger()

__all__ = ["DateOutcome", "download_fleet", "fetch_one_date"]

#: Concurrent Trino queries. **One**, and the quota is not the reason — the
#: account permits 2 running plus 2 queued. Two queries of this shape do not
#: merely contend, they deadlock each other:
#:
#:     20190111 + 20190112, 40 aircraft each, same code path
#:       sequential : 79 s + 41 s = 120 s, both complete
#:       concurrent : > 500 s, zero rows returned by either
#:
#: Not a proportional slowdown — nothing came back at all, where a lone query
#: streams its first rows within 5 s. The cluster grants both a slot, then
#: neither obtains the scan resources the other is holding. Raising this above 1
#: reproduces the 30-minute ``EXCEEDED_TIME_LIMIT`` failures that cost a night's
#: run: 3 dates out of 2,545 in 16 h, against 9 s for one of the same dates run
#: alone.
DEFAULT_WORKERS = 1

#: Aircraft per request. 40 was chosen when the ``EXCEEDED_TIME_LIMIT`` failures
#: were misread as a chunk-size problem; they were a concurrency problem (see
#: :data:`DEFAULT_WORKERS`). Run alone, the sizes behave as one would hope:
#:
#:     20190110, a date that had been failing after 30 min under concurrency
#:        3 aircraft :   9 s
#:       40 aircraft : 111 s, 928,456 rows
#:
#: Measured on 20190109 (192 aircraft), one query at a time:
#:
#:     n= 40 : 111 s,   928,456 rows,  8,365 rows/s,  2.78 s/aircraft
#:     n=100 : 266 s, 2,170,921 rows,  8,153 rows/s,  2.66 s/aircraft
#:     n=200 : 510 s, 4,189,360 rows,  8,208 rows/s,  2.55 s/aircraft
#:
#: Throughput is flat at ~8,200 rows/s whatever the width: the cost is shipping
#: and deserialising rows, not opening a query or scanning a partition. Widening
#: only amortises the ~5 s per-request overhead, worth 8% between 40 and 200 —
#: and the row total is invariant either way, so ~70 h of transfer is the floor
#: for 97,658 aircraft-days however this is tuned.
#:
#: 100 takes most of that gain while keeping a query near 4.5 min against Trino's
#: 30 min ceiling. 200 would buy a further 4% and put the densest day (642
#: aircraft, ~27 min in one chunk) right against the wall that cost a night's run.
CHUNK = 100

#: Smallest chunk :func:`_fetch_chunked` will split down to. Below this, a failure
#: is the query's own problem rather than its size, and halving further would turn
#: one bad date into hundreds of doomed requests.
MIN_CHUNK = 5

#: Substrings identifying a refusal caused by the request being too large. These
#: are NOT retried as-is — the same query would fail identically — but retried
#: with the aircraft list split in half.
_TOO_LARGE = (
    "EXCEEDED_MEMORY_LIMIT",
    "EXCEEDED_LOCAL_MEMORY_LIMIT",
    "EXCEEDED_GLOBAL_MEMORY_LIMIT",
    "EXCEEDED_TIME_LIMIT",
    "EXCEEDED_SCAN_LIMIT",
    "QUERY_TEXT_TOO_LARGE",
    "MAX_LENGTH",
)

#: Retries for a query the cluster refused on capacity grounds. Being told "queue
#: full" means try later, not give up: the earlier code treated it as a permanent
#: failure and dropped the whole date, which on a run of 2,545 dates would discard
#: real work every time a neighbouring query overstayed. Backoff is exponential
#: with jitter so two workers refused at the same instant do not retry in lockstep.
QUEUE_RETRIES = 6
QUEUE_BACKOFF_S = 20.0

#: Requests allowed in flight at once, across every worker.
#:
#: ``ThreadPoolExecutor(max_workers=N)`` bounds *threads*, which is not the same
#: thing. A thread that is sleeping out a ``QUERY_QUEUE_FULL`` backoff still holds
#: its slot, so the pool starts another date — and the observed effect was four
#: dates in flight against a two-query quota, the retries piling onto the very
#: queue they were waiting for. This semaphore is released around the sleep, so
#: what it bounds is queries at the cluster, not threads in the pool.

#: Substrings identifying a refusal that is worth retrying. Matched on the message
#: because ``traffic`` re-raises Trino's error as a plain ``RuntimeError``.
#: ``ADMINISTRATIVELY_KILLED`` is here because a query cancelled from the Trino
#: console — clearing a stuck queue, say — says nothing about whether it could
#: have succeeded. Treating it as terminal drops a date that a later attempt
#: would fetch in seconds; one was lost that way on 20190121.
_RETRYABLE = (
    "QUERY_QUEUE_FULL",
    "INSUFFICIENT_RESOURCES",
    "TOO_MANY_REQUESTS",
    "ADMINISTRATIVELY_KILLED",
)

_KINDS: tuple[_raw_cache.Kind, ...] = ("history", "extended", "flightlist")

# `traffic.data.opensky` resolves lazily and is shared across workers; the import
# itself is not thread-safe, so it is forced once before the pool starts.
_opensky_lock = threading.Lock()
_opensky: Any = None


def _get_opensky() -> Any:
    global _opensky
    with _opensky_lock:
        if _opensky is None:
            from traffic.data import opensky

            _opensky = opensky
    return _opensky


def _is_retryable(exc: Exception) -> bool:
    """Is this a capacity refusal — worth retrying the *same* query later?

    Checked **after** :func:`_is_too_large`, and the order is load-bearing. Trino
    reports a query that ran past its time budget as
    ``INSUFFICIENT_RESOURCES / EXCEEDED_TIME_LIMIT``: the type says "capacity",
    the name says "too big". Matching on the type first would retry an identical
    query that cannot succeed — six times, then fail the date — instead of
    splitting it. So a size marker disqualifies a refusal from being retryable.
    """
    text = str(exc)
    if _is_too_large(exc):
        return False
    return any(marker in text for marker in _RETRYABLE)


def _is_too_large(exc: Exception) -> bool:
    """Did the cluster refuse this because the request itself was too big?"""
    text = str(exc)
    return any(marker in text for marker in _TOO_LARGE)


def _fetch_with_retry(
    fetcher: Any, start: datetime, end: datetime, icao: list[str], *, date_str: str
) -> Any:
    """Call *fetcher*, waiting out capacity refusals.

    Raises:
        Exception: The last refusal, if the cluster is still full after
            :data:`QUEUE_RETRIES` attempts, or immediately for any error that is
            not a capacity refusal.
    """
    for attempt in range(1, QUEUE_RETRIES + 1):
        try:
            # The pipeline raw store is the sole durable cache. pyopensky's
            # query cache duplicates the payload without helping resume.
            return fetcher(start, end, icao24=icao, cached=False)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == QUEUE_RETRIES:
                raise
            delay = QUEUE_BACKOFF_S * (2 ** (attempt - 1)) * (0.5 + random.random())  # noqa: S311
            log.warning(
                "fleet_queue_full",
                date=date_str,
                attempt=f"{attempt}/{QUEUE_RETRIES}",
                sleep_s=round(delay, 1),
            )
        time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class DateOutcome:
    """What one date cost and produced."""

    date: str
    requested: int
    written: int
    requests: int
    empty_kinds: tuple[str, ...]
    error: str | None = None


def _missing_by_owner(
    plan: FleetPlan, kind: _raw_cache.Kind, date_str: str, aircraft: list[str], *, force: bool
) -> dict[str, Cohort]:
    """Aircraft still needed for *kind* on *date*, mapped to their owning cohort."""
    missing: dict[str, Cohort] = {}
    for icao24 in aircraft:
        cohort = plan.owner[icao24]
        if force or not _raw_cache.is_cached(cohort.cfg, kind, date_str, icao24):
            missing[icao24] = cohort
    return missing


def _dispatch_history(
    pdf: Any, wanted: dict[str, Cohort], kind: _raw_cache.Kind, date_str: str
) -> int:
    """Split a per-aircraft result and write each slice into its owner's silo."""
    import polars as pl

    written = 0
    has_column = hasattr(pdf, "columns") and "icao24" in pdf.columns
    for icao24, cohort in wanted.items():
        sub = pdf[pdf["icao24"] == icao24] if has_column else pdf
        frame = pl.from_pandas(sub) if not isinstance(sub, pl.DataFrame) else sub
        _raw_cache.write_atomic(_raw_cache.cache_path(cohort.cfg, kind, date_str, icao24), frame)
        written += 1
    return written


def _collect_flightlist(
    pdf: Any, wanted: dict[str, Cohort], into: dict[str, tuple[Cohort, list[Any]]]
) -> None:
    """Accumulate this chunk's flightlist rows per owning silo.

    ``flightlist`` is cached per date with **no icao24 in its path**, so unlike
    ``history``/``extended`` its slices cannot be written as they arrive: a second
    chunk touching the same cohort would overwrite the first one's file rather than
    add to it. Rows are therefore gathered here and written once per date by
    :func:`_write_flightlist`.

    The per-cohort filter is what keeps the silos isolated — the mutualised result
    covers every cohort flying that day, and writing it unfiltered would give each
    cohort the other 21 cohorts' flights.
    """
    import polars as pl

    frame = pdf if isinstance(pdf, pl.DataFrame) else pl.from_pandas(pdf)
    by_cohort: dict[str, tuple[Cohort, list[str]]] = {}
    for icao24, cohort in wanted.items():
        by_cohort.setdefault(cohort.name, (cohort, []))[1].append(icao24)

    for name, (cohort, aircraft) in by_cohort.items():
        subset = (
            frame.filter(pl.col("icao24").is_in(aircraft)) if "icao24" in frame.columns else frame
        )
        if subset.height:
            into.setdefault(name, (cohort, []))[1].append(subset)


def _write_flightlist(collected: dict[str, tuple[Cohort, list[Any]]], date_str: str) -> int:
    """Write one flightlist parquet per silo from the accumulated chunks."""
    import polars as pl

    written = 0
    for cohort, frames in collected.values():
        merged = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")
        _raw_cache.write_atomic(
            _raw_cache.cache_path(cohort.cfg, "flightlist", date_str, "_"), merged
        )
        written += 1
    return written


def fetch_one_date(plan: FleetPlan, date_str: str, *, force: bool = False) -> DateOutcome:
    """Fetch every kind for one date, mutualised, and dispatch into the silos."""
    aircraft = plan.dates[date_str]
    start = datetime.strptime(date_str, "%Y%m%d")
    end = start + timedelta(hours=24)

    api = _get_opensky()
    written = requests = 0
    empty: list[str] = []

    try:
        for kind in _KINDS:
            wanted = _missing_by_owner(plan, kind, date_str, aircraft, force=force)
            if not wanted:
                continue

            icao_list = sorted(wanted)
            fetcher = api.flightlist if kind == "flightlist" else getattr(api, kind)
            # flightlist has no icao24 in its cache path, so its slices are merged
            # across chunks and written once, after the loop.
            flightlist_rows: dict[str, tuple[Cohort, list[Any]]] = {}

            # A stack, not a for-loop: a chunk the cluster judges too large is
            # replaced by its two halves and retried, so the effective size adapts
            # to what this particular day's traffic allows.
            pending: list[list[str]] = [
                icao_list[i : i + CHUNK] for i in range(0, len(icao_list), CHUNK)
            ]
            while pending:
                chunk = pending.pop()
                try:
                    result = _fetch_with_retry(fetcher, start, end, chunk, date_str=date_str)
                except Exception as exc:
                    if not _is_too_large(exc) or len(chunk) <= MIN_CHUNK:
                        raise
                    half = len(chunk) // 2
                    log.warning(
                        "fleet_chunk_split",
                        date=date_str,
                        kind=kind,
                        n=len(chunk),
                        into=(half, len(chunk) - half),
                    )
                    pending.extend((chunk[:half], chunk[half:]))
                    continue

                requests += 1
                if result is None:
                    empty.append(kind)
                    continue

                chunk_wanted = {a: wanted[a] for a in chunk}
                pdf = result.data if hasattr(result, "data") else result
                if kind == "flightlist":
                    _collect_flightlist(pdf, chunk_wanted, flightlist_rows)
                else:
                    written += _dispatch_history(pdf, chunk_wanted, kind, date_str)

            if flightlist_rows:
                written += _write_flightlist(flightlist_rows, date_str)
    except Exception as exc:  # noqa: BLE001 - one bad date must not kill the fleet run
        return DateOutcome(
            date=date_str,
            requested=len(aircraft),
            written=written,
            requests=requests,
            empty_kinds=tuple(empty),
            error=f"{type(exc).__name__}: {exc}"[:200],
        )

    return DateOutcome(
        date=date_str,
        requested=len(aircraft),
        written=written,
        requests=requests,
        empty_kinds=tuple(empty),
    )


def _append_manifest(manifest_path: Any | None, event: dict[str, Any]) -> None:
    """Append *event* when campaign auditing is enabled."""
    if manifest_path is None:
        return
    from node_fdm_pipeline.commands._fleet_manifest import append_event

    append_event(manifest_path, event)


def _plan_digest(plan: FleetPlan) -> str:
    """Hash the ordered aircraft-day plan recorded by the campaign manifest."""
    import hashlib

    digest = hashlib.sha256()
    for date in sorted(plan.dates):
        aircraft = plan.dates[date]
        digest.update(date.encode())
        digest.update(b"\0")
        for icao24 in sorted(aircraft):
            digest.update(icao24.encode())
            digest.update(b"\0")
    return digest.hexdigest()


def _date_event(outcome: DateOutcome) -> dict[str, Any]:
    """Serialize the stable, public part of a date outcome."""
    return {
        "event": "date_finished",
        "date": outcome.date,
        "requested": outcome.requested,
        "written": outcome.written,
        "requests": outcome.requests,
        "empty_kinds": list(outcome.empty_kinds),
        "error": outcome.error,
    }


def _log_date_outcome(outcome: DateOutcome, done: int, total: int) -> None:
    """Emit one progress record at the appropriate severity."""
    if outcome.error:
        log.error("fleet_date_failed", date=outcome.date, error=outcome.error)
        return
    log.info(
        "fleet_date_done",
        date=outcome.date,
        aircraft=outcome.requested,
        written=outcome.written,
        requests=outcome.requests,
        progress=f"{done}/{total}",
    )


def _run_summary(outcomes: list[DateOutcome]) -> tuple[list[DateOutcome], dict[str, Any]]:
    """Return failed dates and the terminal manifest event."""
    failed = [outcome for outcome in outcomes if outcome.error]
    event: dict[str, Any] = {
        "event": "run_finished",
        "dates": len(outcomes),
        "failed": len(failed),
        "written": sum(outcome.written for outcome in outcomes),
        "requests": sum(outcome.requests for outcome in outcomes),
    }
    return failed, event


def download_fleet(  # noqa: PLR0913
    plan: FleetPlan,
    *,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    dry_run: bool = False,
    manifest_path: Any | None = None,
    fleet_config: FleetRunConfig | None = None,
    recorded_digest: ResumeDigest | None = None,
    selection_digest: str | None = None,
    resolved_config: DigestInput | None = None,
    profile: DigestInput | None = None,
    journal_path: Path | None = None,
    receipt_dir: Path | None = None,
    fetch_boundary: Callable[..., DateOutcome] | None = None,
    acquisition_preflight: AcquisitionPreflight | None = None,
) -> list[DateOutcome]:
    """Download every date in *plan* through one strictly sequential stream."""
    legacy_preflight_supplied = any(
        value is not None
        for value in (
            fleet_config,
            recorded_digest,
            selection_digest,
            resolved_config,
            profile,
        )
    )
    if acquisition_preflight is None and legacy_preflight_supplied:
        if (
            fleet_config is None
            or recorded_digest is None
            or selection_digest is None
            or resolved_config is None
            or profile is None
        ):
            raise ValueError("fleet acquisition preflight inputs are required")
        campaign_root = manifest_path.parent if isinstance(manifest_path, Path) else None
        acquisition_preflight = preflight_acquisition(
            recorded_digest=recorded_digest,
            selection_digest=selection_digest,
            resolved_config=resolved_config,
            profile=profile,
            fleet_config=fleet_config,
            campaign_root=campaign_root,
        )
    if acquisition_preflight is not None and fleet_config is None:
        raise ValueError("fleet_config is required with an acquisition preflight")

    run_key = (
        acquisition_preflight.resume_digest.composite
        if acquisition_preflight is not None
        else _plan_digest(plan)
    )

    if workers != 1:
        raise ValueError(
            "download-fleet is deliberately sequential; --workers must be 1 "
            "because two Trino queries time out instead of increasing throughput"
        )

    per_cohort, mutualised = plan.requests_saved()
    log.info(
        "fleet_plan",
        cohorts=len(plan.cohorts),
        dates=len(plan.dates),
        aircraft_days=plan.aircraft_days,
        requests_per_cohort=per_cohort,
        requests_mutualised=mutualised,
        workers=workers,
    )
    log.info("fleet_planned_dates", dates=sorted(plan.dates))
    if dry_run:
        log.info("fleet_dry_run", msg="Plan valid, would download")
        return []

    _append_manifest(
        manifest_path,
        {
            "event": "run_started",
            "cohorts": len(plan.cohorts),
            "dates": len(plan.dates),
            "aircraft_days": plan.aircraft_days,
            "plan_sha256": run_key,
            "force": force,
        },
    )

    fetch_date = fetch_boundary or fetch_one_date
    outcomes: list[DateOutcome] = []
    done = 0
    total = len(plan.dates)
    acquisition = (
        acquisition_section(
            acquisition_preflight.lease_path,
            owner=run_key,
            ttl_s=fleet_config.lease_ttl_s,
            journal_path=journal_path,
            receipt_dir=receipt_dir,
            acquisition_key=run_key
            if journal_path is not None and receipt_dir is not None
            else None,
        )
        if acquisition_preflight is not None and fleet_config is not None
        else nullcontext()
    )
    with acquisition:
        _get_opensky()  # resolve the lazy import before the remote span starts
        for date_str in plan.dates:
            outcome = fetch_date(plan, date_str, force=force)
            outcomes.append(outcome)
            done += 1
            _append_manifest(manifest_path, _date_event(outcome))
            _log_date_outcome(outcome, done, total)

    failed, summary = _run_summary(outcomes)
    log.info(
        "fleet_done",
        dates=summary["dates"],
        failed=summary["failed"],
        written=summary["written"],
        requests=summary["requests"],
    )
    if failed:
        log.warning("fleet_failed_dates", dates=[o.date for o in failed][:20])
    _append_manifest(manifest_path, summary)
    return outcomes
