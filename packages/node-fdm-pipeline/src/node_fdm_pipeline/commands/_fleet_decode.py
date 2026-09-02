"""Decode every cohort's raw cache into its Delta table, several at a time.

``download-fleet`` deliberately does not chain ``decode``. It is date-major: each
date it fetches feeds ~14 silos at once, so no cohort is complete until the last
date of the plan is done, and there is nothing to decode incrementally. Decoding
is therefore a separate phase, and it is a different kind of work:

    download  is bounded by the Trino quota (2 concurrent queries) and is almost
              all network wait.
    decode    makes zero network calls. It is CPU and memory bound, so it scales
              with cores instead of with a remote quota.

The limit here is RAM, and it is severe enough that the default is **one worker**.
A full decode of A330-800 — the *smallest* of the 22 cohorts — peaks near 94 GiB,
and the memory is anonymous heap, not reclaimable page cache. The largest cohort
is 1.68x that, so a single decode can be expected to approach 160 GiB and two
concurrent ones would want ~316 GiB on a 250 GiB host.

An earlier default of 6 came from a run where only one cohort had a raw cache, so
five workers exited in seconds having found nothing to do; the peak that run
recorded described one decode, not six. Sizing a pool from a measurement where
the pool was never full is the mistake this note exists to prevent.

Processes, not threads: each decode wants its own interpreter for the CPU work,
and its own address space so one cohort's peak does not stack onto another's
inside a single heap.

The pool is started with **spawn**, not the platform default. On Linux that
default is ``fork``, which copies the parent's memory but only the calling thread:
a lock another thread happened to hold is inherited locked, and nothing in the
child will ever release it. That is not theoretical here — a caller sampling
``/proc`` from a background thread deadlocked every worker in ``futex_do_wait``
with zero CPU consumed, looking exactly like a slow decode. ``spawn`` starts each
worker from a clean interpreter, which costs a second of import time per worker
and removes the whole class of failure.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from node_fdm_pipeline.commands._fleet_boundary import (
    AcquisitionPreflight,
    FleetGuardDecision,
    acquisition_section,
    preflight_acquisition,
    resolve_fleet_guard,
)
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigest,
    record_campaign_identity,
)
from node_fdm_pipeline.config import FleetRunConfig

log = structlog.get_logger()

__all__ = [
    "DEFAULT_WORKERS",
    "CohortDecode",
    "DecodeOutcome",
    "decode_fleet",
    "plan_decodes",
]

#: Concurrent decode processes. One, because a single decode of the smallest
#: cohort already peaks near 94 GiB and the largest is 1.68x that. This is a
#: memory budget, not a parallelism target — the host has 48 cores and they are
#: not the constraint. Raise it only against a measured peak for the cohorts
#: actually being decoded, and only if that peak times the worker count fits.
DEFAULT_WORKERS = 1

#: Peak anonymous heap of one decode, GiB. Measured on A330-800 (3,206
#: aircraft-days, 116M rows); the largest cohort is 1.68x. Used to warn before a
#: run rather than discover the ceiling through the OOM killer.
DECODE_RSS_GIB = 95


@dataclass(frozen=True)
class CohortDecode:
    """One cohort's decode job: its config and the span its selection covers."""

    name: str
    config_path: Path
    start_date: str
    end_date: str
    days: int


@dataclass(frozen=True)
class DecodeOutcome:
    """What one cohort's decode produced."""

    name: str
    seconds: float
    error: str | None = None


def _available_gib() -> float:
    """Memory a new process can actually claim, in GiB.

    ``/proc/meminfo``'s ``MemAvailable``, not ``SC_AVPHYS_PAGES``: the latter counts
    only free pages and ignores reclaimable page cache, which on a host that has
    just written tens of gigabytes of parquet is most of the usable memory. Here
    the two read 72 GiB and 222 GiB — the difference decides whether 6 workers look
    safe or impossible, so the pessimistic figure would refuse a run that fits.

    Falls back to ``SC_AVPHYS_PAGES`` where ``/proc`` is unreadable.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2**30
    except (OSError, IndexError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
    except (OSError, ValueError):
        pages = os.sysconf("SC_PHYS_PAGES")
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**30


type _CampaignInputs = tuple[
    FleetRunConfig | None,
    ResumeDigest | None,
    str | None,
    DigestInput | None,
    DigestInput | None,
]


def _resolve_decode_guard(
    decision: FleetGuardDecision | None,
    acquisition_preflight: AcquisitionPreflight | None,
    campaign: _CampaignInputs,
) -> tuple[FleetGuardDecision, AcquisitionPreflight | None]:
    """Resolve historical or campaign execution without nesting the entrypoint."""
    if decision is not None:
        return decision, acquisition_preflight

    fleet_config, recorded_digest, selection_digest, resolved_config, profile = campaign
    if (
        fleet_config is None
        or recorded_digest is None
        or selection_digest is None
        or resolved_config is None
        or profile is None
    ):
        return (
            resolve_fleet_guard(
                selection=None,
                resolved_config=None,
                profile=None,
                lease_path=None,
            ),
            acquisition_preflight,
        )

    preflight = preflight_acquisition(
        recorded_digest=recorded_digest,
        selection_digest=selection_digest,
        resolved_config=resolved_config,
        profile=profile,
        fleet_config=fleet_config,
    )
    return (
        FleetGuardDecision(
            mode="campaign",
            preflight=preflight,
            resume_digest=preflight.resume_digest,
        ),
        preflight,
    )


def decode_fleet(  # noqa: PLR0913
    fleet_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
    fleet_config: FleetRunConfig | None = None,
    recorded_digest: ResumeDigest | None = None,
    selection_digest: str | None = None,
    resolved_config: DigestInput | None = None,
    profile: DigestInput | None = None,
    journal_path: Path | None = None,
    receipt_dir: Path | None = None,
    decision: FleetGuardDecision | None = None,
    acquisition_preflight: AcquisitionPreflight | None = None,
    resume_digest_path: Path | None = None,
) -> list[DecodeOutcome]:
    """Decode a fleet after validating its resume identity and shared lease."""
    guard, preflight = _resolve_decode_guard(
        decision,
        acquisition_preflight,
        (fleet_config, recorded_digest, selection_digest, resolved_config, profile),
    )

    if guard.mode == "historical":
        return _decode_fleet_impl(fleet_dir, workers=workers, dry_run=dry_run)
    if fleet_config is None or preflight is None:
        raise ValueError("campaign decode requires fleet configuration and preflight")
    if dry_run:
        return _decode_fleet_impl(fleet_dir, workers=workers, dry_run=True)

    run_key = preflight.resume_digest.composite
    with acquisition_section(
        preflight.lease_path,
        owner=run_key,
        ttl_s=fleet_config.lease_ttl_s,
        journal_path=journal_path,
        receipt_dir=receipt_dir,
        acquisition_key=run_key,
    ):
        if resume_digest_path is not None:
            _record_resume_digest(resume_digest_path, preflight.resume_digest)
        return _decode_fleet_impl(fleet_dir, workers=workers, dry_run=False)


def _record_resume_digest(path: Path, digest: ResumeDigest) -> None:
    """Atomically record the campaign identity while its shared lease is held."""
    record_campaign_identity(path.parent, digest)


def plan_decodes(fleet_dir: Path) -> list[CohortDecode]:
    """Derive each cohort's decode span from its own selection.

    The span is per cohort, not global: a cohort whose selection starts in 2020
    has nothing cached for 2019, and handing ``decode`` the fleet-wide range would
    make it walk — and log a skip for — hundreds of empty days.

    The end date is the last selected day **plus one**, matching how ``download``
    derives its range (``data.py``), because ``decode`` walks ``while current <
    end`` and would otherwise drop the final day.
    """
    from node_fdm_pipeline.commands._fleet_plan import build_fleet_plan, discover_cohorts

    plan = build_fleet_plan(discover_cohorts(fleet_dir))
    jobs: list[CohortDecode] = []
    for cohort in plan.cohorts:
        days = sorted(
            date
            for date, aircraft in plan.dates.items()
            if any(cohort in plan.owner[a] for a in aircraft)
        )
        if not days:
            log.warning("decode_plan_empty", cohort=cohort.name)
            continue
        last = datetime.strptime(days[-1], "%Y%m%d") + timedelta(days=1)
        jobs.append(
            CohortDecode(
                name=cohort.name,
                config_path=cohort.config_path,
                start_date=f"{days[0][:4]}-{days[0][4:6]}-{days[0][6:]}",
                end_date=last.strftime("%Y-%m-%d"),
                days=len(days),
            )
        )
    return jobs


def _run_one(job: CohortDecode) -> DecodeOutcome:
    """Decode one cohort in this worker process."""
    import time

    from node_fdm_pipeline.commands.data import decode

    t0 = time.perf_counter()
    try:
        decode(
            config=job.config_path,
            start_date=job.start_date,
            end_date=job.end_date,
        )
    except Exception as exc:  # noqa: BLE001 - one cohort must not kill the fleet run
        return DecodeOutcome(
            name=job.name,
            seconds=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )
    return DecodeOutcome(name=job.name, seconds=time.perf_counter() - t0)


def _log_decode_plan(jobs: list[CohortDecode], workers: int) -> None:
    """Log the scheduled work and warn when its memory estimate is tight."""
    budget_gib = workers * DECODE_RSS_GIB
    log.info(
        "decode_fleet_plan",
        cohorts=len(jobs),
        cohort_days=sum(job.days for job in jobs),
        workers=workers,
        est_peak_gib=budget_gib,
    )
    available_gib = _available_gib()
    if budget_gib > available_gib:
        log.warning(
            "decode_fleet_memory_tight",
            est_peak_gib=budget_gib,
            available_gib=round(available_gib),
            msg="lower --workers or decode in two passes",
        )


def _log_dry_run(jobs: list[CohortDecode]) -> None:
    """Log every job that a dry run would execute."""
    for job in jobs:
        log.info(
            "decode_fleet_would_run",
            cohort=job.name,
            start=job.start_date,
            end=job.end_date,
            days=job.days,
        )


def _execute_decodes(jobs: list[CohortDecode], workers: int) -> list[DecodeOutcome]:
    """Run planned jobs in spawned worker processes."""
    outcomes: list[DecodeOutcome] = []
    done = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            done += 1
            if outcome.error:
                log.error("decode_fleet_failed", cohort=outcome.name, error=outcome.error)
                continue
            log.info(
                "decode_fleet_done",
                cohort=outcome.name,
                seconds=round(outcome.seconds, 1),
                progress=f"{done}/{len(jobs)}",
            )
    return outcomes


def _log_decode_summary(outcomes: list[DecodeOutcome]) -> None:
    """Log aggregate decode results and failed cohort names."""
    failed = [outcome for outcome in outcomes if outcome.error]
    log.info(
        "decode_fleet_complete",
        cohorts=len(outcomes),
        failed=len(failed),
        seconds=round(sum(outcome.seconds for outcome in outcomes), 1),
    )
    if failed:
        log.warning("decode_fleet_failed_cohorts", cohorts=[outcome.name for outcome in failed])


def _decode_fleet_impl(
    fleet_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
) -> list[DecodeOutcome]:
    """Decode every cohort under *fleet_dir*, *workers* at a time."""
    jobs = plan_decodes(fleet_dir)
    # Largest first: the long tail of small cohorts then fills the workers that
    # finish early, instead of one big cohort running alone at the end.
    jobs.sort(key=lambda job: job.days, reverse=True)

    _log_decode_plan(jobs, workers)
    if dry_run:
        _log_dry_run(jobs)
        return []

    outcomes = _execute_decodes(jobs, workers)
    _log_decode_summary(outcomes)
    return outcomes
