from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from node_fdm_pipeline.commands._campaign_guard import CampaignMode, resolve_campaign_mode
from node_fdm_pipeline.commands._campaign_guard import preflight_campaign as _preflight_campaign
from node_fdm_pipeline.commands._campaign_plan import (
    CampaignPlan,
    _build_campaign_fleet_plan,
    campaign_plan,
)
from node_fdm_pipeline.commands._campaign_report import (
    CampaignReport,
    campaign_status,
    campaign_validate,
)
from node_fdm_pipeline.commands._campaign_runner import (
    CampaignRunReport,
    CampaignStepRun,
)
from node_fdm_pipeline.commands._day_plan import DayPlan, build_day_plan
from node_fdm_pipeline.commands._fleet_boundary import preflight_acquisition
from node_fdm_pipeline.commands._fleet_digest import (
    load_campaign_identity,
    record_campaign_identity,
)
from node_fdm_pipeline.commands._fleet_fetch import download_fleet
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan, load_selection_file
from node_fdm_pipeline.commands._science_profile import profile_manifest, resolve_science_profile
from node_fdm_pipeline.config import FleetRunConfig, PipelineConfig

__all__ = ["run_fleet_campaign"]

_SCIENCE_PROFILE = profile_manifest(resolve_science_profile("opensky26-exp03-v1"))


class CampaignExecutionNotReadyError(RuntimeError):
    """Raised instead of reporting success for an unwired live campaign stage."""


class CampaignDownloadFailedError(RuntimeError):
    """Raised when at least one source day fails during acquisition."""


def _resolve_path(config_path: Path, value: Path) -> Path:
    return value if value.is_absolute() else config_path.parent / value


def _live_context(
    config: Path,
    mode: CampaignMode,
) -> tuple[Path, FleetRunConfig, SelectionPlan, Path, Path]:
    config_path = config.expanduser().resolve()
    resolved = PipelineConfig.from_yaml(config_path)
    fleet_run = resolved.fleet_run
    if fleet_run is None:
        raise ValueError("campaign execution requires fleet_run configuration")
    if fleet_run.recorded_source is None:
        raise ValueError("campaign execution requires fleet_run.recorded_source")
    if fleet_run.acquisition_journal is None:
        raise ValueError("campaign execution requires fleet_run.acquisition_journal")
    if fleet_run.acquisition_receipt_dir is None:
        raise ValueError("campaign execution requires fleet_run.acquisition_receipt_dir")

    selection_path = _resolve_path(config_path, fleet_run.recorded_source)
    source = load_selection_file(selection_path)
    if source.plan is None:
        raise ValueError("recorded campaign selection must be structured")

    journal_path = _resolve_path(config_path, fleet_run.acquisition_journal)
    receipt_dir = _resolve_path(config_path, fleet_run.acquisition_receipt_dir)
    digest = _preflight_campaign(
        mode=mode,
        state_dir=config_path.parent,
        selection_digest=source.plan.digest,
        resolved_config=fleet_run,
        profile=_SCIENCE_PROFILE,
    )
    if mode is CampaignMode.RUN:
        record_campaign_identity(config_path.parent, digest)
    return config_path, fleet_run, source.plan, journal_path, receipt_dir


def _day_plans(selection: SelectionPlan) -> tuple[DayPlan, ...]:
    days = sorted({day for flight in selection.flights for day in flight.utc_days})
    return tuple(build_day_plan(selection, day) for day in days)


def _run_live(
    config: Path,
    mode: CampaignMode,
    steps: tuple[str, ...],
) -> CampaignRunReport:
    _config_path, fleet_run, selection, journal_path, receipt_dir = _live_context(config, mode)
    if steps == ("download",):
        return _run_download_only(
            _config_path,
            fleet_run,
            selection,
            journal_path,
            receipt_dir,
        )
    raise CampaignExecutionNotReadyError(
        "live decode/enrich orchestration is not wired yet; "
        "use --only-step download for the acquisition-only campaign"
    )


def _run_download_only(
    config_path: Path,
    fleet_run: FleetRunConfig,
    selection: SelectionPlan,
    journal_path: Path,
    receipt_dir: Path,
) -> CampaignRunReport:
    """Run the real, strictly sequential acquisition engine for one campaign."""
    resolved = PipelineConfig.from_yaml(config_path)
    runtime_fleet_run = fleet_run
    if fleet_run.recorded_opensky_source is not None:
        runtime_fleet_run = fleet_run.model_copy(
            update={
                "recorded_opensky_source": _resolve_path(
                    config_path,
                    fleet_run.recorded_opensky_source,
                )
            }
        )
    assert fleet_run.recorded_source is not None
    selection_path = _resolve_path(config_path, fleet_run.recorded_source)
    fleet_plan = _build_campaign_fleet_plan(
        config_path,
        resolved,
        selection,
        selection_path,
    )
    acquisition_preflight = preflight_acquisition(
        recorded_digest=load_campaign_identity(config_path.parent),
        selection_digest=selection.digest,
        resolved_config=fleet_run,
        profile=_SCIENCE_PROFILE,
        fleet_config=runtime_fleet_run,
        campaign_root=config_path.parent,
    )
    outcomes = download_fleet(
        fleet_plan,
        workers=1,
        force=False,
        dry_run=False,
        manifest_path=journal_path,
        fleet_config=runtime_fleet_run,
        journal_path=journal_path,
        receipt_dir=receipt_dir,
        acquisition_preflight=acquisition_preflight,
    )
    failures = tuple(outcome for outcome in outcomes if outcome.error is not None)
    if failures:
        failed_days = ", ".join(outcome.date for outcome in failures)
        raise CampaignDownloadFailedError(f"OpenSky acquisition failed for: {failed_days}")

    days = tuple(plan.meta_selection_day for plan in _day_plans(selection))
    return CampaignRunReport(
        started_days=days,
        completed_days=days,
        blocked_reasons=(),
        blocking_day=None,
        max_observed_concurrency=1,
        steps=(CampaignStepRun(step="download"),),
    )


_CAMPAIGN_STEPS = ("download", "decode", "enrich")


class UnknownCampaignStep(ValueError):  # noqa: N818 - public contract name
    """Raised when a requested step is outside the public campaign contract."""

    def __init__(self, step: str) -> None:
        accepted = ", ".join(_CAMPAIGN_STEPS)
        super().__init__(f"unknown campaign step {step!r}; accepted steps: {accepted}")


def _resolve_campaign_steps(only_steps: Sequence[str] | None) -> tuple[str, ...]:
    selected = _CAMPAIGN_STEPS if only_steps is None else only_steps
    requested = tuple(dict.fromkeys(selected))
    for step in requested:
        if step not in _CAMPAIGN_STEPS:
            raise UnknownCampaignStep(step)
    return requested


def run_fleet_campaign(
    config: Path,
    mode: str,
    *,
    only_steps: Sequence[str] | None = None,
) -> CampaignPlan | CampaignReport | CampaignRunReport:
    """Run exactly one public campaign contract for the requested mode."""
    steps = _resolve_campaign_steps(only_steps)
    campaign_mode = resolve_campaign_mode(mode)
    if campaign_mode is CampaignMode.PLAN:
        return campaign_plan(config)
    if campaign_mode is CampaignMode.STATUS:
        return campaign_status(config)
    if campaign_mode is CampaignMode.VALIDATE:
        return campaign_validate(config)
    return _run_live(config, campaign_mode, steps)
