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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from node_fdm_pipeline.commands import _cache_retention, _raw_cache
from node_fdm_pipeline.commands._fetch_backoff import BackoffPolicy, next_delay, retry_policy_for
from node_fdm_pipeline.commands._fetch_bisect import plan_bisection
from node_fdm_pipeline.commands._fleet_boundary import (
    AcquisitionPreflight,
    acquisition_section,
    preflight_acquisition,
)
from node_fdm_pipeline.commands._fleet_digest import DigestInput, ResumeDigest
from node_fdm_pipeline.commands._fleet_journal import AttemptRecord, replay_attempts
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._trino_errors import classify_trino_failure
from node_fdm_pipeline.config import FleetRunConfig

if TYPE_CHECKING:
    from node_fdm_pipeline.commands._fleet_plan import Cohort, FleetPlan

log = structlog.get_logger()

__all__ = ["DateOutcome", "download_fleet", "fetch_action", "fetch_one_date"]

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


def fetch_action(error: BaseException) -> str:
    """Return the typed acquisition action for a Trino failure."""
    return classify_trino_failure(error).kind


def _is_retryable(exc: Exception) -> bool:
    """Is this a capacity refusal — worth retrying the *same* query later?

    Checked **after** :func:`_is_too_large`, and the order is load-bearing. Trino
    reports a query that ran past its time budget as
    ``INSUFFICIENT_RESOURCES / EXCEEDED_TIME_LIMIT``: the type says "capacity",
    the name says "too big". Matching on the type first would retry an identical
    query that cannot succeed — six times, then fail the date — instead of
    splitting it. So a size marker disqualifies a refusal from being retryable.
    """
    return fetch_action(exc) == "retry_later"


def _is_too_large(exc: Exception) -> bool:
    """Did the cluster refuse this because the request itself was too big?"""
    return fetch_action(exc) == "split"


def _default_backoff_policy() -> BackoffPolicy:
    return BackoffPolicy(
        min_delay_s=QUEUE_BACKOFF_S / 2,
        max_delay_s=QUEUE_BACKOFF_S * (2 ** (QUEUE_RETRIES - 1)) * 1.5,
        max_retries=QUEUE_RETRIES,
        base_s=QUEUE_BACKOFF_S,
    )


def _backoff_policy(config: FleetRunConfig | None) -> BackoffPolicy:
    if config is None:
        return _default_backoff_policy()
    return BackoffPolicy(
        min_delay_s=config.retry_min_delay_s,
        max_delay_s=config.retry_max_delay_s,
        max_retries=config.retry_max_retries,
        base_s=config.retry_base_s,
    )


def _attempt_event(  # noqa: PLR0913
    *,
    date_str: str,
    kind: _raw_cache.Kind,
    batch: list[str],
    attempt: int,
    outcome: str,
    observed_delay_s: float,
    acquisition_keys: tuple[str, ...] = (),
) -> dict[str, object]:
    batch_label = _batch_label(batch)
    return {
        "event": "fetch_attempt",
        "attempt_id": f"{date_str}:{kind}:{batch_label}:{attempt}",
        "date": date_str,
        "kind": kind,
        "batch": batch,
        "batch_label": batch_label,
        "attempt": attempt,
        "outcome": outcome,
        "observed_delay_s": observed_delay_s,
        "branch_name": batch_label,
        "acquisition_keys": acquisition_keys,
    }


def _fetch_with_retry(  # noqa: PLR0913
    fetcher: Any,
    start: datetime,
    end: datetime,
    icao: list[str],
    *,
    date_str: str,
    kind: _raw_cache.Kind = "history",
    manifest_path: object | None = None,
    policy: BackoffPolicy | None = None,
    rng: random.Random | None = None,
    acquisition_keys: tuple[str, ...] = (),
) -> Any:
    """Call *fetcher*, waiting out capacity refusals.

    Raises:
        Exception: The last refusal, if the cluster is still full after
            :data:`QUEUE_RETRIES` attempts, or immediately for any error that is
            not a capacity refusal.
    """
    resolved_policy = policy or _default_backoff_policy()
    resolved_rng = rng or random.Random()  # noqa: S311 - retry jitter is not cryptographic
    attempt = (
        _next_attempt_rank(
            manifest_path,
            date_str=date_str,
            kind=kind,
            batch=icao,
        )
        - 1
    )
    while True:
        attempt += 1
        try:
            # The pipeline raw store is the sole durable cache. pyopensky's
            # query cache duplicates the payload without helping resume.
            result = fetcher(start, end, icao24=icao, cached=False)
        except Exception as exc:
            failure = classify_trino_failure(exc)
            if failure.kind == "split":
                raise
            decision = retry_policy_for(failure, resolved_policy)
            outcome = "retry_later" if decision.allowed else "terminal"
            delay = (
                next_delay(attempt=attempt, policy=resolved_policy, rng=resolved_rng)
                if decision.allowed
                else 0.0
            )
            _append_manifest(
                manifest_path,
                _attempt_event(
                    date_str=date_str,
                    kind=kind,
                    batch=icao,
                    attempt=attempt,
                    outcome=outcome,
                    observed_delay_s=delay,
                    acquisition_keys=acquisition_keys,
                ),
            )
            if not decision.allowed or attempt >= decision.max_attempts:
                raise
            log.warning(
                "fleet_queue_full",
                date=date_str,
                attempt=f"{attempt}/{decision.max_attempts}",
                sleep_s=round(delay, 1),
            )
            time.sleep(delay)
            continue
        _append_manifest(
            manifest_path,
            _attempt_event(
                date_str=date_str,
                kind=kind,
                batch=icao,
                attempt=attempt,
                outcome="success",
                observed_delay_s=0.0,
                acquisition_keys=acquisition_keys,
            ),
        )
        return result


@dataclass(frozen=True)
class DateOutcome:
    """What one date cost and produced."""

    date: str
    requested: int
    written: int
    requests: int
    empty_kinds: tuple[str, ...]
    error: str | None = None
    kind: str = "success"
    failing_batch: tuple[str, ...] = ()
    staged_icao24s: tuple[str, ...] = ()
    staged_digest: str | None = None


def _missing_by_owner(
    plan: FleetPlan, kind: _raw_cache.Kind, date_str: str, aircraft: list[str], *, force: bool
) -> dict[str, tuple[Cohort, ...]]:
    """Aircraft still needed for *kind* on *date*, mapped to their owning cohort."""
    missing: dict[str, tuple[Cohort, ...]] = {}
    for icao24 in aircraft:
        owner = plan.owner[icao24]
        owner_cohorts = owner if isinstance(owner, tuple) else (owner,)
        owners = (
            owner_cohorts
            if force or not _raw_cache.is_cached(owner_cohorts[0].cfg, kind, date_str, icao24)
            else ()
        )
        if owners:
            missing[icao24] = owners
    return missing


def _dispatch_history(
    pdf: Any, wanted: dict[str, Cohort], kind: _raw_cache.Kind, date_str: str
) -> int:
    """Split a result and stage each aircraft slice once for all consumers."""
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


@dataclass(frozen=True)
class _KindFetch:
    plan: FleetPlan
    kind: _raw_cache.Kind
    date: str
    aircraft: list[str]
    start: datetime
    end: datetime
    force: bool
    fetcher: Callable[..., object]
    manifest_path: object | None = None
    backoff_policy: BackoffPolicy | None = None
    bisection_floor: int = MIN_CHUNK


@dataclass
class _KindCounters:
    written: int = 0
    requests: int = 0
    empty: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ChunkFetch:
    result: object | None = None
    split: tuple[list[str], list[str]] | None = None


def _branch_event(
    context: _KindFetch, parent: list[str], child: tuple[str, ...]
) -> dict[str, object]:
    return {
        "event": "fetch_branch",
        "date": context.date,
        "kind": context.kind,
        "batch": list(child),
        "parent_batch": parent,
    }


@runtime_checkable
class _TerminalFloorFailure(Protocol):
    kind: str
    batch: tuple[str, ...]


@runtime_checkable
class _TerminalFloorProgress(Protocol):
    written: int
    requests: int
    staged_icao24s: tuple[str, ...]
    staged_digest: str | None


class _TerminalFloorError(RuntimeError):
    kind = "terminal_floor"

    def __init__(self, batch: list[str], cause: Exception) -> None:
        super().__init__(str(cause))
        self.batch = tuple(batch)
        self.written = 0
        self.requests = 0
        self.staged_icao24s: tuple[str, ...] = ()
        self.staged_digest: str | None = None


def _selection_acquisition_keys(
    plan: FleetPlan,
    date_str: str,
    batch: list[str],
) -> tuple[str, ...]:
    if getattr(plan, "selection", None) is None:
        return ()

    keys: list[str] = []
    for icao24 in batch:
        selected = plan.resolve(icao24, date_str)
        if selected is not None:
            keys.append(selected.acquisition_key)
    return tuple(keys)


def _fetch_or_split(context: _KindFetch, chunk: list[str]) -> _ChunkFetch:
    acquisition_keys = _selection_acquisition_keys(context.plan, context.date, chunk)
    try:
        result = _fetch_with_retry(
            context.fetcher,
            context.start,
            context.end,
            chunk,
            date_str=context.date,
            kind=context.kind,
            manifest_path=context.manifest_path,
            policy=context.backoff_policy,
            acquisition_keys=acquisition_keys,
        )
    except Exception as exc:
        if fetch_action(exc) != "split":
            raise
        plan = plan_bisection(chunk, floor=context.bisection_floor)
        _append_manifest(
            context.manifest_path,
            _attempt_event(
                date_str=context.date,
                kind=context.kind,
                batch=chunk,
                attempt=1,
                outcome=plan.kind,
                observed_delay_s=0.0,
                acquisition_keys=acquisition_keys,
            ),
        )
        if plan.kind == "terminal_floor":
            raise _TerminalFloorError(chunk, exc) from exc
        for child in plan.children:
            _append_manifest(context.manifest_path, _branch_event(context, chunk, child))
        left, right = plan.children
        log.warning(
            "fleet_chunk_split",
            date=context.date,
            kind=context.kind,
            n=len(chunk),
            into=(len(left), len(right)),
        )
        return _ChunkFetch(split=(list(left), list(right)))
    return _ChunkFetch(result=result)


def _dispatch_chunk(
    context: _KindFetch,
    chunk: list[str],
    wanted: dict[str, tuple[Cohort, ...]],
    result: object,
    flightlist_rows: dict[str, tuple[Cohort, list[object]]],
) -> int:
    pdf = getattr(result, "data", result)
    if context.kind == "flightlist":
        by_cohort: dict[str, dict[str, Cohort]] = {}
        for aircraft in chunk:
            for cohort in wanted[aircraft]:
                by_cohort.setdefault(cohort.name, {})[aircraft] = cohort
        for cohort_wanted in by_cohort.values():
            _collect_flightlist(pdf, cohort_wanted, flightlist_rows)
        return 0

    shared_wanted = {aircraft: wanted[aircraft][0] for aircraft in chunk}
    written = _dispatch_history(pdf, shared_wanted, context.kind, context.date)
    consumers = {cohort.name for owner_cohorts in wanted.values() for cohort in owner_cohorts}
    representative = next(iter(shared_wanted.values()))
    _raw_cache.write_consumer_ledger(
        representative.cfg,
        context.kind,
        context.date,
        consumers,
    )
    return written


def _staged_payload_digest(
    context: _KindFetch,
    staged_icao24s: list[str],
) -> str | None:
    if not staged_icao24s:
        return None

    import hashlib

    digest = hashlib.sha256()
    representative = context.plan.cohorts[0]
    for icao24 in staged_icao24s:
        path = _raw_cache.cache_path(
            representative.cfg,
            context.kind,
            context.date,
            icao24,
        )
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fetch_kind(context: _KindFetch) -> _KindCounters:
    wanted = _missing_by_owner(
        context.plan,
        context.kind,
        context.date,
        context.aircraft,
        force=context.force,
    )
    counters = _KindCounters()
    if not wanted:
        return counters

    aircraft = sorted(wanted)
    pending = [aircraft[index : index + CHUNK] for index in range(0, len(aircraft), CHUNK)]
    flightlist_rows: dict[str, tuple[Cohort, list[object]]] = {}
    staged_icao24s: list[str] = []
    terminal_error: _TerminalFloorError | None = None
    while pending:
        chunk = pending.pop()
        try:
            fetched = _fetch_or_split(context, chunk)
        except _TerminalFloorError as exc:
            if terminal_error is None:
                terminal_error = exc
            continue
        if fetched.split is not None:
            pending.extend(reversed(fetched.split))
            continue
        counters.requests += 1
        if fetched.result is None:
            counters.empty.append(context.kind)
            continue
        written = _dispatch_chunk(
            context,
            chunk,
            wanted,
            fetched.result,
            flightlist_rows,
        )
        counters.written += written
        if written:
            staged_icao24s.extend(chunk)

    if flightlist_rows:
        counters.written += _write_flightlist(flightlist_rows, context.date)
    if terminal_error is not None:
        terminal_error.written = counters.written
        terminal_error.requests = counters.requests
        terminal_error.staged_icao24s = tuple(staged_icao24s)
        terminal_error.staged_digest = _staged_payload_digest(context, staged_icao24s)
        raise terminal_error
    return counters


@dataclass(frozen=True)
class _DateOutcomeInput:
    date: str
    aircraft: list[str]
    written: int
    requests: int
    empty: list[str]
    error: Exception | None = None


def _date_outcome(values: _DateOutcomeInput) -> DateOutcome:
    error = values.error
    detail = None if error is None else f"{type(error).__name__}: {error}"[:200]
    kind = "success" if error is None else "error"
    failing_batch: tuple[str, ...] = ()
    written = values.written
    requests = values.requests
    staged_icao24s: tuple[str, ...] = ()
    staged_digest: str | None = None

    if isinstance(error, _TerminalFloorFailure) and error.kind == "terminal_floor":
        kind = "terminal_floor"
        failing_batch = error.batch
    if isinstance(error, _TerminalFloorProgress):
        written = error.written
        requests = error.requests
        staged_icao24s = error.staged_icao24s
        staged_digest = error.staged_digest

    return DateOutcome(
        date=values.date,
        requested=len(values.aircraft),
        written=written,
        requests=requests,
        empty_kinds=tuple(values.empty),
        error=detail,
        kind=kind,
        failing_batch=failing_batch,
        staged_icao24s=staged_icao24s,
        staged_digest=staged_digest,
    )


def _batch_label(batch: list[str]) -> str:
    """Return the stable journal label for one submitted batch."""
    if not batch:
        return "empty"
    return f"{batch[0]}-{batch[-1]}"


def _recorded_attempts(manifest_path: object | None) -> tuple[AttemptRecord, ...]:
    """Read typed attempts when a durable manifest is available."""
    if not isinstance(manifest_path, (str, Path)):
        return ()
    path = Path(manifest_path)
    if not path.exists():
        return ()
    return replay_attempts(read_events(path))


def _next_attempt_rank(
    manifest_path: object | None,
    *,
    date_str: str,
    kind: _raw_cache.Kind,
    batch: list[str],
) -> int:
    """Continue after the last durable rank for the same logical batch."""
    recorded_ranks = (
        event.attempt
        for event in _recorded_attempts(manifest_path)
        if event.date == date_str and event.kind == kind and event.batch == tuple(batch)
    )
    return max(recorded_ranks, default=0) + 1


def _remaining_aircraft(
    aircraft: list[str],
    *,
    date_str: str,
    kind: _raw_cache.Kind,
    manifest_path: object | None,
) -> list[str]:
    """Remove successfully received aircraft before any source submission."""
    received = {
        icao24
        for event in _recorded_attempts(manifest_path)
        if event.date == date_str and event.kind == kind and event.outcome == "success"
        for icao24 in event.batch
    }
    return [icao24 for icao24 in aircraft if icao24 not in received]


def fetch_one_date(
    plan: FleetPlan,
    date_str: str,
    *,
    force: bool = False,
    manifest_path: object | None = None,
    fleet_config: FleetRunConfig | None = None,
) -> DateOutcome:
    """Fetch every kind for one date and stage the mutualised payloads."""
    aircraft = plan.dates[date_str]
    start = datetime.strptime(date_str, "%Y%m%d")
    end = start + timedelta(hours=24)
    api = _get_opensky()
    written = requests = 0
    empty: list[str] = []

    try:
        for kind in _KINDS:
            pending_aircraft = _remaining_aircraft(
                aircraft,
                date_str=date_str,
                kind=kind,
                manifest_path=manifest_path,
            )
            if not pending_aircraft:
                continue
            fetcher = api.flightlist if kind == "flightlist" else getattr(api, kind)
            counters = _fetch_kind(
                _KindFetch(
                    plan,
                    kind,
                    date_str,
                    pending_aircraft,
                    start,
                    end,
                    force,
                    fetcher,
                    manifest_path,
                    _backoff_policy(fleet_config),
                    fleet_config.bisection_floor if fleet_config is not None else MIN_CHUNK,
                )
            )
            written += counters.written
            requests += counters.requests
            empty.extend(counters.empty)
    except Exception as exc:  # noqa: BLE001 - one bad date must not kill the fleet run
        return _date_outcome(_DateOutcomeInput(date_str, aircraft, written, requests, empty, exc))

    return _date_outcome(_DateOutcomeInput(date_str, aircraft, written, requests, empty))


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
        "kind": getattr(outcome, "kind", "success"),
        "failing_batch": list(getattr(outcome, "failing_batch", ())),
        "staged_icao24s": list(getattr(outcome, "staged_icao24s", ())),
        "staged_digest": getattr(outcome, "staged_digest", None),
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


@dataclass(frozen=True)
class _PreflightInputs:
    fleet_config: FleetRunConfig | None
    recorded_digest: ResumeDigest | None
    selection_digest: str | None
    resolved_config: DigestInput | None
    profile: DigestInput | None
    manifest_path: object | None


@dataclass(frozen=True)
class _DownloadContext:
    plan: FleetPlan
    force: bool
    manifest_path: object | None
    fleet_config: FleetRunConfig | None
    preflight: AcquisitionPreflight | None
    journal_path: Path | None
    receipt_dir: Path | None
    fetch_date: Callable[..., DateOutcome]


def _preflight_supplied(inputs: _PreflightInputs) -> bool:
    return any(
        value is not None
        for value in (
            inputs.fleet_config,
            inputs.recorded_digest,
            inputs.selection_digest,
            inputs.resolved_config,
            inputs.profile,
        )
    )


def _required[T](value: T | None) -> T:
    if value is None:
        raise ValueError("fleet acquisition preflight inputs are required")
    return value


def _build_preflight(inputs: _PreflightInputs) -> AcquisitionPreflight:
    campaign_root = inputs.manifest_path.parent if isinstance(inputs.manifest_path, Path) else None
    return preflight_acquisition(
        recorded_digest=_required(inputs.recorded_digest),
        selection_digest=_required(inputs.selection_digest),
        resolved_config=_required(inputs.resolved_config),
        profile=_required(inputs.profile),
        fleet_config=_required(inputs.fleet_config),
        campaign_root=campaign_root,
    )


def _resolve_preflight(
    current: AcquisitionPreflight | None,
    inputs: _PreflightInputs,
) -> AcquisitionPreflight | None:
    if current is None and _preflight_supplied(inputs):
        current = _build_preflight(inputs)
    if current is not None and inputs.fleet_config is None:
        raise ValueError("fleet_config is required with an acquisition preflight")
    return current


def _download_run_key(plan: FleetPlan, preflight: AcquisitionPreflight | None) -> str:
    if preflight is None:
        return _plan_digest(plan)
    return preflight.resume_digest.composite


def _validate_workers(workers: int) -> None:
    if workers == 1:
        return
    raise ValueError(
        "download-fleet is deliberately sequential; --workers must be 1 "
        "because two Trino queries time out instead of increasing throughput"
    )


def _log_plan(plan: FleetPlan, workers: int) -> None:
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


def _run_started_event(plan: FleetPlan, run_key: str, force: bool) -> dict[str, object]:
    return {
        "event": "run_started",
        "cohorts": len(plan.cohorts),
        "dates": len(plan.dates),
        "aircraft_days": plan.aircraft_days,
        "plan_sha256": run_key,
        "force": force,
    }


def _campaign_pending_dates(plan: FleetPlan, *, force: bool) -> set[str]:
    if force:
        return set(plan.dates)
    try:
        cfg = plan.cohorts[0].cfg
    except (AttributeError, IndexError):
        return set(plan.dates)

    pending: set[str] = set()
    for day, icao24s in plan.dates.items():
        for kind in _KINDS:
            root = _raw_cache.cache_root(cfg, kind)
            reconciliation = _cache_retention.reconcile_day(
                root,
                day,
                kind,
                icao24s=icao24s,
            )
            if reconciliation.status == "invalid":
                _cache_retention.invalidate_day(root, day, kind)
                pending.add(day)
    return pending


def _run_dates(context: _DownloadContext, run_key: str) -> list[DateOutcome]:
    outcomes: list[DateOutcome] = []
    pending_dates = _campaign_pending_dates(context.plan, force=context.force)
    acquisition = (
        acquisition_section(
            context.preflight.lease_path,
            owner=run_key,
            ttl_s=context.fleet_config.lease_ttl_s,
            journal_path=context.journal_path,
            receipt_dir=context.receipt_dir,
            acquisition_key=run_key
            if context.journal_path is not None and context.receipt_dir is not None
            else None,
        )
        if context.preflight is not None and context.fleet_config is not None
        else nullcontext()
    )
    with acquisition:
        if pending_dates:
            _get_opensky()
        for done, date_str in enumerate(context.plan.dates, start=1):
            if date_str in pending_dates:
                outcome = context.fetch_date(context.plan, date_str, force=context.force)
            else:
                outcome = DateOutcome(
                    date=date_str,
                    requested=len(context.plan.dates[date_str]),
                    written=0,
                    requests=0,
                    empty_kinds=(),
                )
            outcomes.append(outcome)
            _append_manifest(context.manifest_path, _date_event(outcome))
            _log_date_outcome(outcome, done, len(context.plan.dates))
    return outcomes


def _finish_run(manifest_path: object | None, outcomes: list[DateOutcome]) -> None:
    failed, summary = _run_summary(outcomes)
    log.info(
        "fleet_done",
        dates=summary["dates"],
        failed=summary["failed"],
        written=summary["written"],
        requests=summary["requests"],
    )
    if failed:
        log.warning("fleet_failed_dates", dates=[outcome.date for outcome in failed][:20])
    _append_manifest(manifest_path, summary)


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
    acquisition_preflight = _resolve_preflight(
        acquisition_preflight,
        _PreflightInputs(
            fleet_config,
            recorded_digest,
            selection_digest,
            resolved_config,
            profile,
            manifest_path,
        ),
    )
    run_key = _download_run_key(plan, acquisition_preflight)
    _validate_workers(workers)
    _log_plan(plan, workers)
    if dry_run:
        log.info("fleet_dry_run", msg="Plan valid, would download")
        return []

    _append_manifest(manifest_path, _run_started_event(plan, run_key, force))
    outcomes = _run_dates(
        _DownloadContext(
            plan,
            force,
            manifest_path,
            fleet_config,
            acquisition_preflight,
            journal_path,
            receipt_dir,
            fetch_boundary
            or partial(
                fetch_one_date,
                manifest_path=manifest_path,
                fleet_config=fleet_config,
            ),
        ),
        run_key,
    )
    _finish_run(manifest_path, outcomes)
    return outcomes
