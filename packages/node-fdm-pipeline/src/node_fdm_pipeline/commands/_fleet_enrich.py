"""Date-major, resumable ERA5 enrichment for every fleet cohort."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from node_fdm_pipeline.commands._fleet_boundary import (
    AcquisitionPreflight,
    FleetGuardDecision,
    acquisition_section,
    preflight_acquisition,
    resolve_fleet_guard,
)
from node_fdm_pipeline.commands._fleet_digest import DigestInput, ResumeDigest
from node_fdm_pipeline.commands._fleet_manifest import append_event
from node_fdm_pipeline.commands.data import (
    ENRICH_INPUT_COLUMNS,
    enrich_with_grid,
    validate_enriched_frame,
)
from node_fdm_pipeline.commands.data import (
    EnrichOutcome as EnrichOutcome,
)
from node_fdm_pipeline.config import FleetRunConfig, PipelineConfig

log = structlog.get_logger()

_ERA_COLUMNS = (
    "era_temp_K",
    "era_u_wind_ms",
    "era_v_wind_ms",
    "era_tas_kt",
    "era_mach",
    "era_cas_kt",
)
_COMPACT_DAY_LENGTH = 8


@dataclass(frozen=True)
class EnrichmentCohort:
    """One decoded cohort participating in the weather campaign."""

    name: str
    cfg: PipelineConfig


@dataclass(frozen=True)
class EnrichmentPlan:
    """Cohorts grouped by the actual dates present in their Delta tables."""

    days: dict[str, tuple[EnrichmentCohort, ...]]
    cache_root: Path
    features: tuple[str, ...]

    @property
    def assignments(self) -> int:
        """Number of (date, cohort) enrichment writes in the plan."""
        return sum(len(cohorts) for cohorts in self.days.values())


@dataclass(frozen=True)
class DateEnrichmentOutcome:
    """Terminal state for one calendar day."""

    day: str
    enriched: int
    skipped: int
    cache_purged: bool
    error: str | None = None


ProviderFactory = Callable[[Path, tuple[str, ...]], Any]
StatusChecker = Callable[[EnrichmentCohort, str], tuple[bool, int, float]]
EnrichFunction = Callable[..., EnrichOutcome]


@dataclass(frozen=True)
class EnrichmentRuntime:
    """Injectable weather boundaries used by the campaign runner."""

    provider_factory: ProviderFactory
    status_checker: StatusChecker
    enrich_function: EnrichFunction


def _normalise_day(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    if len(text) == _COMPACT_DAY_LENGTH and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    return date.fromisoformat(text).isoformat()


def _table_days(table_path: Path) -> tuple[str, ...]:
    import polars as pl

    if not table_path.exists():
        raise SystemExit(f"decoded Delta table missing: {table_path}")
    try:
        frame = pl.scan_delta(str(table_path)).select("meta_batch_date").unique().collect()
    except Exception as exc:
        raise SystemExit(f"cannot inspect decoded Delta table {table_path}: {exc}") from exc
    return tuple(sorted(_normalise_day(value) for value in frame["meta_batch_date"].drop_nulls()))


def _bounded(days: Iterable[str], start_date: str, end_date: str) -> tuple[str, ...]:
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    return tuple(
        day
        for day in days
        if (start is None or date.fromisoformat(day) >= start)
        and (end is None or date.fromisoformat(day) < end)
    )


def enrich_fleet(  # noqa: PLR0913
    plan: EnrichmentPlan,
    *,
    dry_run: bool = False,
    manifest_path: Path | None = None,
    runtime: EnrichmentRuntime | None = None,
    fleet_config: FleetRunConfig | None = None,
    recorded_digest: ResumeDigest | None = None,
    selection_digest: str | None = None,
    resolved_config: DigestInput | None = None,
    profile: DigestInput | None = None,
    journal_path: Path | None = None,
    receipt_dir: Path | None = None,
    decision: FleetGuardDecision | None = None,
    acquisition_preflight: AcquisitionPreflight | None = None,
) -> list[DateEnrichmentOutcome]:
    """Enrich a fleet after validating its resume identity and shared lease."""
    guard = decision
    preflight = acquisition_preflight
    legacy_values = (
        fleet_config,
        recorded_digest,
        selection_digest,
        resolved_config,
        profile,
    )
    if guard is None:
        if all(value is None for value in legacy_values):
            guard = resolve_fleet_guard(
                selection=None,
                resolved_config=None,
                profile=None,
                lease_path=None,
            )
        elif all(value is not None for value in legacy_values):
            assert fleet_config is not None
            assert recorded_digest is not None
            assert selection_digest is not None
            assert resolved_config is not None
            assert profile is not None
            campaign_root = manifest_path.parent if manifest_path is not None else None
            preflight = preflight_acquisition(
                recorded_digest=recorded_digest,
                selection_digest=selection_digest,
                resolved_config=resolved_config,
                profile=profile,
                fleet_config=fleet_config,
                campaign_root=campaign_root,
            )
            guard = FleetGuardDecision(
                mode="campaign",
                preflight=preflight,
                resume_digest=preflight.resume_digest,
            )
        else:
            raise ValueError("fleet acquisition preflight inputs must be provided together")

    if guard.mode == "historical":
        return _enrich_fleet_impl(
            plan,
            dry_run=dry_run,
            manifest_path=manifest_path,
            runtime=runtime,
        )
    if fleet_config is None or preflight is None:
        raise ValueError("campaign enrich requires fleet configuration and preflight")
    if dry_run:
        return _enrich_fleet_impl(
            plan,
            dry_run=True,
            manifest_path=manifest_path,
            runtime=runtime,
        )

    run_key = preflight.resume_digest.composite
    with acquisition_section(
        preflight.lease_path,
        owner=run_key,
        ttl_s=fleet_config.lease_ttl_s,
        journal_path=journal_path,
        receipt_dir=receipt_dir,
        acquisition_key=run_key,
    ):
        return _enrich_fleet_impl(
            plan,
            dry_run=False,
            manifest_path=manifest_path,
            runtime=runtime,
        )


def build_enrichment_plan(
    triples: list[tuple[str, Path, Path]],
    *,
    data_root: Path | None = None,
    start_date: str = "",
    end_date: str = "",
    day_reader: Callable[[Path], tuple[str, ...]] | None = None,
) -> EnrichmentPlan:
    """Build a date-major plan from decoded data, never from census estimates."""
    by_day: dict[str, list[EnrichmentCohort]] = defaultdict(list)
    cache_roots: set[Path] = set()
    feature_sets: set[tuple[str, ...]] = set()

    read_days = day_reader or _table_days
    for name, config_path, _selection_path in triples:
        cfg = PipelineConfig.from_yaml(config_path, data_root=data_root)
        cohort = EnrichmentCohort(name=name, cfg=cfg)
        cache_roots.add(cfg.paths.resolve("era5_cache_dir").resolve())
        feature_sets.add(tuple(sorted(cfg.era5_features)))
        for day in _bounded(read_days(cfg.paths.resolve("delta_table")), start_date, end_date):
            by_day[day].append(cohort)

    if not by_day:
        raise SystemExit("no decoded cohort dates found in the requested window")
    if len(cache_roots) != 1:
        raise SystemExit("fleet configs do not resolve to one shared ERA5 cache root")
    if len(feature_sets) != 1:
        raise SystemExit("fleet configs request different ERA5 feature sets")

    return EnrichmentPlan(
        days={
            day: tuple(sorted(cohorts, key=lambda cohort: cohort.name))
            for day, cohorts in sorted(by_day.items())
        },
        cache_root=next(iter(cache_roots)),
        features=next(iter(feature_sets)),
    )


def _default_provider(cache_path: Path, features: tuple[str, ...]) -> Any:
    try:
        from fastmeteo.source.arco_era5 import ArcoEra5
    except ImportError:
        raise RuntimeError(
            "fastmeteo is required for ERA5 enrichment; install the weather extra"
        ) from None
    kwargs: dict[str, object] = {"local_store": str(cache_path)}
    if features:
        kwargs["features"] = list(features)
    return ArcoEra5(**kwargs)


def _close_provider(provider: Any | None) -> None:
    if provider is None:
        return
    remote = getattr(provider, "remote_dataset", None)
    close = getattr(remote, "close", None)
    if callable(close):
        close()


def weather_status(cohort: EnrichmentCohort, day: str) -> tuple[bool, int, float]:
    import polars as pl

    table_path = cohort.cfg.paths.resolve("delta_table")
    scan = pl.scan_delta(str(table_path))
    schema = scan.collect_schema().names()
    if any(column not in schema for column in _ERA_COLUMNS):
        return False, 0, 0.0
    frame = (
        scan.filter(pl.col("meta_batch_date") == day.replace("-", ""))
        .select([*ENRICH_INPUT_COLUMNS, *_ERA_COLUMNS])
        .collect()
    )
    if frame.is_empty():
        return False, 0, 0.0
    try:
        outcome = validate_enriched_frame(frame, cohort.cfg.era5_null_threshold)
    except RuntimeError:
        return False, len(frame), 0.0
    return True, outcome.rows, outcome.max_null_fraction


def _purge_day_cache(cache_root: Path, day_cache: Path) -> bool:
    root = cache_root.resolve()
    target = day_cache.resolve()
    expected_name = f"date={_normalise_day(target.name.removeprefix('date='))}"
    if target.parent != root or target.name != expected_name:
        raise RuntimeError(f"refusing unsafe ERA5 cache purge: {target}")
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _plan_digest(plan: EnrichmentPlan) -> str:
    payload = {
        "days": {day: [cohort.name for cohort in cohorts] for day, cohorts in plan.days.items()},
        "features": plan.features,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _next_day(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def _record(path: Path | None, event: dict[str, object]) -> None:
    if path is not None:
        append_event(path, event)


def _close_day_provider(
    provider: Any | None,
    day: str,
    error: str | None,
    manifest_path: Path | None,
) -> str | None:
    try:
        _close_provider(provider)
    except Exception as exc:  # noqa: BLE001 - a failed close must retain the day cache
        close_error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            log.warning("era5_provider_close_failed", date=day, error=close_error)
            return error
        _record(
            manifest_path,
            {
                "event": "date_failed",
                "date": day,
                "error": close_error,
                "cache_retained": True,
            },
        )
        return close_error
    return error


def _purge_validated_cache(
    cache_root: Path,
    day_cache: Path,
    day: str,
    manifest_path: Path | None,
) -> tuple[bool, str | None]:
    try:
        return _purge_day_cache(cache_root, day_cache), None
    except Exception as exc:  # noqa: BLE001 - purge failure is a resumable day failure
        error = f"{type(exc).__name__}: {exc}"
        _record(
            manifest_path,
            {
                "event": "date_failed",
                "date": day,
                "error": error,
                "cache_retained": day_cache.exists(),
            },
        )
        return False, error


def _run_day(
    plan: EnrichmentPlan,
    day: str,
    cohorts: tuple[EnrichmentCohort, ...],
    *,
    runtime: EnrichmentRuntime,
    manifest_path: Path | None,
) -> DateEnrichmentOutcome:
    day_cache = plan.cache_root / f"date={day}"
    provider: Any | None = None
    enriched = 0
    skipped = 0
    error: str | None = None
    _record(manifest_path, {"event": "date_started", "date": day, "cohorts": len(cohorts)})

    try:
        for cohort in cohorts:
            complete, rows, null_fraction = runtime.status_checker(cohort, day)
            if complete:
                skipped += 1
                _record(
                    manifest_path,
                    {
                        "event": "cohort_skipped",
                        "date": day,
                        "cohort": cohort.name,
                        "rows": rows,
                        "max_null_fraction": null_fraction,
                        "reason": "already_complete",
                    },
                )
                continue

            if provider is None:
                day_cache.mkdir(parents=True, exist_ok=True)
                provider = runtime.provider_factory(day_cache, plan.features)
            outcome = runtime.enrich_function(
                cohort.cfg,
                provider,
                start_date=day,
                end_date=_next_day(day),
            )
            complete, rows, null_fraction = runtime.status_checker(cohort, day)
            if not complete or rows != outcome.rows:
                raise RuntimeError(f"post-write ERA5 validation failed for {cohort.name} on {day}")
            enriched += 1
            _record(
                manifest_path,
                {
                    "event": "cohort_finished",
                    "date": day,
                    "cohort": cohort.name,
                    "rows": rows,
                    "max_null_fraction": null_fraction,
                },
            )
    except Exception as exc:  # noqa: BLE001 - campaign boundary records partial failure
        error = f"{type(exc).__name__}: {exc}"
        _record(
            manifest_path,
            {
                "event": "date_failed",
                "date": day,
                "error": error,
                "cache_retained": True,
            },
        )
    finally:
        error = _close_day_provider(provider, day, error, manifest_path)

    if error is not None:
        return DateEnrichmentOutcome(
            day=day,
            enriched=enriched,
            skipped=skipped,
            cache_purged=False,
            error=error,
        )

    purged, error = _purge_validated_cache(plan.cache_root, day_cache, day, manifest_path)
    if error is not None:
        return DateEnrichmentOutcome(
            day=day,
            enriched=enriched,
            skipped=skipped,
            cache_purged=False,
            error=error,
        )
    _record(
        manifest_path,
        {
            "event": "date_finished",
            "date": day,
            "enriched": enriched,
            "skipped": skipped,
            "cache_purged": purged,
        },
    )
    return DateEnrichmentOutcome(
        day=day,
        enriched=enriched,
        skipped=skipped,
        cache_purged=purged,
    )


def _enrich_fleet_impl(
    plan: EnrichmentPlan,
    *,
    dry_run: bool = False,
    manifest_path: Path | None = None,
    runtime: EnrichmentRuntime | None = None,
) -> list[DateEnrichmentOutcome]:
    """Enrich chronologically; advance only after a fully validated day."""
    log.info(
        "enrich_fleet_plan",
        dates=len(plan.days),
        assignments=plan.assignments,
        cache_root=str(plan.cache_root),
    )
    if dry_run:
        return [
            DateEnrichmentOutcome(day=day, enriched=0, skipped=len(cohorts), cache_purged=False)
            for day, cohorts in plan.days.items()
        ]

    active_runtime = runtime or EnrichmentRuntime(
        provider_factory=_default_provider,
        status_checker=weather_status,
        enrich_function=enrich_with_grid,
    )
    _record(
        manifest_path,
        {
            "event": "run_started",
            "plan_sha256": _plan_digest(plan),
            "dates": len(plan.days),
            "assignments": plan.assignments,
        },
    )
    outcomes: list[DateEnrichmentOutcome] = []
    for day, cohorts in plan.days.items():
        outcome = _run_day(
            plan,
            day,
            cohorts,
            runtime=active_runtime,
            manifest_path=manifest_path,
        )
        outcomes.append(outcome)
        if outcome.error is not None:
            break

    failed = sum(outcome.error is not None for outcome in outcomes)
    _record(
        manifest_path,
        {
            "event": "run_finished",
            "completed_dates": sum(outcome.error is None for outcome in outcomes),
            "failed": failed,
            "remaining_dates": len(plan.days) - len(outcomes),
        },
    )
    return outcomes
