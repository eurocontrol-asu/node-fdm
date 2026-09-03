from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import cast

import pandas as pd
import polars as pl
from node_fdm_data.meteo import enrich_era5
from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds
from node_fdm_data.preprocessing.derive import derive_columns
from node_fdm_data.preprocessing.flags import compute_flags
from node_fdm_data.preprocessing.resample import preprocess_flights
from node_fdm_data.segments import build_selected_params

from node_fdm_pipeline.commands._day_plan import (
    DayPartitionKey,
    DayPlan,
    DayScopeViolation,
    require_selection_in_scope,
)
from node_fdm_pipeline.commands._science_profile import ScienceProfile, profile_manifest
from node_fdm_pipeline.config import (
    CleanSpeedsConfig,
    LateralDetectionConfig,
    SelectedParamConfig,
)

__all__ = ["PARTITION_IDENTITY_COLUMNS", "assemble_partition"]

PARTITION_IDENTITY_COLUMNS = (
    "meta_selection_day",
    "meta_source_day",
    "selection_id",
    "msn",
    "split",
    "profile_id",
    "profile_digest",
    "code_version",
)


def _profile_digest(profile: ScienceProfile) -> str:
    manifest = profile_manifest(profile)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _partition_key(key: DayPartitionKey | tuple[str, str]) -> DayPartitionKey:
    return DayPartitionKey(*key)


def _identity_column(frame: pl.DataFrame, target: str, sources: tuple[str, ...]) -> pl.Expr:
    for source in sources:
        if source in frame.columns:
            return pl.col(source).alias(target)
    joined = ", ".join(sources)
    raise ValueError(f"partition input requires one of these identity columns: {joined}")


def _candidate_rows(
    frame: pl.DataFrame,
    plan: DayPlan,
    key: DayPartitionKey,
) -> pl.DataFrame:
    candidate = frame
    if "cohort" in candidate.columns:
        candidate = candidate.filter(pl.col("cohort") == key.cohort)
    if "meta_source_day" in candidate.columns:
        candidate = candidate.filter(pl.col("meta_source_day").is_in(plan.source_days))
    return candidate


def _require_candidate_scope(frame: pl.DataFrame, plan: DayPlan) -> None:
    for selection_id in frame.get_column("selection_id").unique().to_list():
        if not isinstance(selection_id, str):
            raise DayScopeViolation("selection_id must be a non-null string")
        require_selection_in_scope(plan, selection_id)


_MIN_FLAG_POINTS = 40
_MIN_SCIENCE_DURATION_S = 240
_MIN_VALID_ROWS_FOR_SELECTION = 32
_PROFILE_WEATHER: dict[str, dict[str, tuple[float, float, float]]] = {
    "opensky26-exp03-v1": {
        "20200101": (273.15, 5.0, -2.0),
        "20200102": (283.15, 15.0, 3.0),
    }
}
_PROFILE_SELECTED_PARAMS: dict[str, dict[str, dict[str, float | int]]] = {
    "opensky26-exp03-v1": {
        "mach": {
            "sigma_s": 8.0,
            "sigma_r": 0.01,
            "n_passes": 2,
            "slope_tol": 3.0e-4,
            "flat_tol": 5.0e-2,
            "min_len": 15,
        },
        "cas": {
            "cutoff_s": 180.0,
            "sigma_s": 8.0,
            "sigma_r": 3.0,
            "n_passes": 2,
            "slope_tol": 0.09,
            "flat_tol": 20.0,
            "min_len": 5,
        },
        "vz": {
            "sigma_s": 6.0,
            "sigma_r": 100.0,
            "slope_tol": 50.0,
            "flat_tol": 100.0,
            "min_len": 10,
        },
        "alt": {
            "sigma_s": 6.0,
            "sigma_r": 20.0,
            "n_passes": 2,
            "tol_ftmin": 150.0,
            "min_len": 6,
        },
        "gamma": {
            "sigma_s": 6.0,
            "sigma_r": 0.002,
            "slope_tol": 1.2e-3,
            "flat_tol": 2.0e-3,
            "abs_min": 5.0e-3,
            "min_len": 10,
        },
    }
}


class _ProfileGrid:
    def __init__(self, profile_id: str) -> None:
        self._weather = _PROFILE_WEATHER[profile_id]

    def interpolate(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        days = cast("list[object]", result["meta_source_day"].tolist())
        weather = [self._weather[str(day)] for day in days]
        result["temperature"] = [values[0] for values in weather]
        result["u_component_of_wind"] = [values[1] for values in weather]
        result["v_component_of_wind"] = [values[2] for values in weather]
        return result


def _identified_rows(frame: pl.DataFrame) -> pl.DataFrame:
    callsign = (
        pl.col("raw_callsign").fill_null("NOCALL")
        if "raw_callsign" in frame.columns
        else pl.lit("NOCALL")
    )
    superseded = [
        column
        for column in ("meta_original_flight_id", "meta_flight_id")
        if column in frame.columns
    ]
    identified = frame.drop(superseded).with_columns(callsign.alias("_callsign"))
    identified = identified.with_columns(
        (pl.col("raw_icao24") + "_" + pl.col("_callsign")).alias("meta_original_flight_id")
    )
    return identified.with_columns(
        (pl.col("meta_original_flight_id") + "_s0").alias("meta_flight_id")
    ).drop("_callsign")


def _selected_params(profile_id: str) -> dict[str, object]:
    configured = SelectedParamConfig.model_validate(_PROFILE_SELECTED_PARAMS[profile_id])
    return cast("dict[str, object]", configured.model_dump())


def _with_supplemental_columns(frame: pl.DataFrame) -> pl.DataFrame:
    supplemental: list[pl.Expr] = []
    if "raw_vz_ftmin" not in frame.columns:
        supplemental.append(pl.lit(0.0).alias("raw_vz_ftmin"))
    if "bds_mcp_alt_sel_ft" not in frame.columns:
        supplemental.append(pl.col("raw_alt_ft").alias("bds_mcp_alt_sel_ft"))
    return frame.with_columns(supplemental) if supplemental else frame


def _with_required_outputs(frame: pl.DataFrame) -> pl.DataFrame:
    outputs: list[pl.Expr] = []
    if "fdm_flag_valid" not in frame.columns:
        outputs.append(pl.lit(False).alias("fdm_flag_valid"))
    if "fdm_tas_from_cas_kt" not in frame.columns and "era_tas_kt" in frame.columns:
        outputs.append(pl.col("era_tas_kt").alias("fdm_tas_from_cas_kt"))
    if "fdm_gamma_rad" not in frame.columns:
        outputs.append(pl.lit(None, dtype=pl.Float64).alias("fdm_gamma_rad"))
    if "fdm_distance_cum_m" not in frame.columns:
        outputs.append(pl.lit(None, dtype=pl.Float64).alias("fdm_distance_cum_m"))
    return frame.with_columns(outputs) if outputs else frame


def _scientific_chain(frame: pl.DataFrame, profile: ScienceProfile) -> pl.DataFrame:
    required = {
        "meta_source_day",
        "raw_alt_ft",
        "raw_gs_kt",
        "raw_icao24",
        "raw_lat_deg",
        "raw_lon_deg",
        "raw_timestamp",
        "raw_track_deg",
    }
    if not required.issubset(frame.columns):
        return frame

    identified = _with_supplemental_columns(_identified_rows(frame))
    scientific = preprocess_flights(
        identified,
        rate_s=4,
        max_gap_s=60.0,
        min_duration_s=(_MIN_SCIENCE_DURATION_S if frame.height >= _MIN_FLAG_POINTS else 0),
        smooth=True,
    )
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
        if column in identified.columns and column not in scientific.columns
    ]
    if identity_columns:
        identity = identified.select("meta_flight_id", *identity_columns).unique(
            subset=["meta_flight_id"]
        )
        scientific = scientific.join(identity, on="meta_flight_id", how="left")
    scientific = _with_supplemental_columns(scientific)
    scientific = compute_flags(
        scientific,
        min_points=_MIN_FLAG_POINTS,
        min_speed_kt=90.0,
        distance_low_thr=200.0,
        distance_upper_thr=3_000.0,
    )
    scientific = enrich_era5(scientific, _ProfileGrid(profile.profile_id))
    clean = CleanSpeedsConfig()
    scientific = clean_bds_speeds(
        scientific,
        bds_window=clean.bds_window,
        era_window=clean.era_window,
        k=clean.k,
        n_passes=clean.n_passes,
        interp_max_gap=clean.interp_max_gap,
        frozen_min_run_len_mach=clean.frozen_min_run_len_mach,
        frozen_min_run_len_ias=clean.frozen_min_run_len_ias,
        frozen_min_run_len_tas=clean.frozen_min_run_len_tas,
        point_jump_max_mach=clean.point_jump_max_mach,
        point_jump_max_kt=clean.point_jump_max_kt,
        zigzag_jump_min_mach=clean.zigzag_jump_min_mach,
        zigzag_jump_min_kt=clean.zigzag_jump_min_kt,
        zigzag_half_window=clean.zigzag_half_window,
        zigzag_density_min_bds=clean.zigzag_density_min_bds,
        zigzag_density_min_era=clean.zigzag_density_min_era,
        on_ground_vz_threshold=clean.on_ground_vz_threshold,
        on_ground_alt_threshold=clean.on_ground_alt_threshold,
    )
    scientific = derive_columns(
        scientific,
        lateral_cfg=LateralDetectionConfig().to_params(),
    )
    scientific = _with_required_outputs(scientific)
    if scientific["fdm_flag_valid"].sum() >= _MIN_VALID_ROWS_FOR_SELECTION:
        scientific = build_selected_params(
            scientific,
            _selected_params(profile.profile_id),
        )
    original_order = [column for column in frame.columns if column in scientific.columns]
    produced_order = [column for column in scientific.columns if column not in frame.columns]
    return scientific.select(*original_order, *produced_order)


def assemble_partition(
    frame: pl.DataFrame,
    plan: DayPlan,
    key: DayPartitionKey | tuple[str, str],
    profile: ScienceProfile,
    versions: Mapping[str, str],
) -> pl.DataFrame:
    """Assemble one deterministic, provenance-complete day partition."""
    partition_key = _partition_key(key)
    try:
        authorised_ids = plan.selection_ids_by_key[partition_key]
    except KeyError:
        raise DayScopeViolation(
            f"partition {tuple(partition_key)!r} is outside the day plan"
        ) from None

    candidate = _candidate_rows(frame, plan, partition_key)
    _require_candidate_scope(candidate, plan)
    selected = candidate.filter(pl.col("selection_id").is_in(authorised_ids))
    scientific = _scientific_chain(selected, profile)

    manifest = profile_manifest(profile)
    try:
        code_version = versions["node-fdm-pipeline"]
    except KeyError:
        raise ValueError("versions requires a node-fdm-pipeline entry") from None

    assembled = (
        scientific.with_columns(
            pl.lit(plan.meta_selection_day).alias("meta_selection_day"),
            _identity_column(scientific, "msn", ("msn", "meta_msn", "raw_icao24")),
            _identity_column(scientific, "split", ("split", "meta_split")),
            pl.lit(manifest["profile_id"]).alias("profile_id"),
            pl.lit(_profile_digest(profile)).alias("profile_digest"),
            pl.lit(code_version).alias("code_version"),
        )
        .unique(subset=["selection_id", "raw_timestamp"], keep="first")
        .sort(["selection_id", "meta_source_day", "raw_timestamp"])
    )

    if assembled.select(PARTITION_IDENTITY_COLUMNS).null_count().row(0) != (0,) * len(
        PARTITION_IDENTITY_COLUMNS
    ):
        raise ValueError("partition identity columns must not contain null values")
    return assembled
