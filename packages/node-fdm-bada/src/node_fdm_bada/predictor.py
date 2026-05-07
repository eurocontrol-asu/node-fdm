"""Predictor for generating BADA 4.2-based flight trajectories.

Wraps EUROCONTROL's pyBADA TCL module to produce reference trajectories
that can be compared with Neural ODE predictions.

.. note::
   ``pyBADA`` is proprietary and not bundled with this package.
   If unavailable, :func:`process_single_flight` logs a warning and
   returns ``None``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog
from node_fdm_data.physics.constants import FT
from node_fdm_data.physics.isa import isa_temperature

from node_fdm_bada.utils import get_phase, ms_to_kt, tas_to_cas

__all__ = [
    "process_single_flight",
]

log = structlog.get_logger()

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# pyBADA conditional import — proprietary dependency
# ---------------------------------------------------------------------------
try:
    from pyBADA.TCL import (
        accDec_time,
        constantSpeedLevel,
        constantSpeedRating_time,
        constantSpeedROCD_time,
        target,
    )

    _HAS_PYBADA = True
except ImportError:
    _HAS_PYBADA = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NM_TO_MS: float = 1852.0 / 3600.0  # 1 kt in m/s
_G: float = 9.80665  # m/s² — standard gravitational acceleration
# Below this threshold, ``fdm_in_turn`` is treated as numerical noise and the
# step is forwarded to pyBADA without ``turnMetrics`` overrides.
_TURN_RATE_THRESHOLD_RADS: float = 1e-3


def _bank_angle_for_turn(*, omega_rads: float, tas_ms: float) -> float:
    """Bank angle (deg) from the coordinated-turn formula ``tan(φ) = ω·V/g``.

    Computed locally to sidestep the pyBADA 0.1.5 ``Airplane.bankAngle``
    AttributeError raised from ``geodesic.py:884`` when ``rateOfTurn`` is set
    but ``bankAngle`` is left as the default zero.
    """
    return float(np.degrees(np.arctan(omega_rads * tas_ms / _G)))


def _turn_direction(d_heading_rads: float) -> str | None:
    """Map a signed heading rate to pyBADA's ``directionOfTurn`` string.

    Heading is measured clockwise (standard nav convention), so a positive
    rate means a right turn.
    """
    if d_heading_rads > 0:
        return "RIGHT"
    if d_heading_rads < 0:
        return "LEFT"
    return None


def _step_turn_metrics(row: dict[str, Any], current_tas: float) -> tuple[float, float, str | None]:
    """Return ``(turn_rate_dps, bank_angle_deg, turn_direction)`` for one step.

    Below ``_TURN_RATE_THRESHOLD_RADS`` (or when ``fdm_in_turn`` is false), the
    defaults ``(0.0, 0.0, None)`` are returned so pyBADA stays in its straight
    flight regime.
    """
    d_heading_rads = float(row.get("fdm_d_heading_rads", 0.0))
    if not (
        bool(row.get("fdm_in_turn", False)) and abs(d_heading_rads) > _TURN_RATE_THRESHOLD_RADS
    ):
        return 0.0, 0.0, None
    return (
        float(np.degrees(abs(d_heading_rads))),
        _bank_angle_for_turn(omega_rads=abs(d_heading_rads), tas_ms=current_tas),
        _turn_direction(d_heading_rads),
    )


def _backfill_cas_sel(df: pl.DataFrame) -> pl.DataFrame:
    cas_sel = df["cas_sel_ms"].to_numpy().copy()
    if "cas_ms" in df.columns:
        cas_sel[-1] = df["cas_ms"][-1]
    cas_sel[cas_sel == 0.0] = np.nan
    for i in range(len(cas_sel) - 2, -1, -1):
        if np.isnan(cas_sel[i]):
            cas_sel[i] = cas_sel[i + 1]
    return df.with_columns(pl.Series("cas_sel_ms", cas_sel))


def _commanded_heading(row: dict[str, Any], current_heading_deg: float | None) -> float | None:
    if bool(row.get("fdm_heading_target_known", False)):
        return float(np.degrees(row["fdm_heading_target_rad"]))
    return current_heading_deg


def process_single_flight(
    flight_path: str | Path,
    ac: Any,
    *,
    processor: Any | None = None,
    output_dir: str | Path | None = None,
) -> pl.DataFrame | None:
    """Process one flight with BADA 4.2 predictions.

    Reads the flight parquet, iterates over time steps, and calls pyBADA
    TCL functions to predict altitude, speed, ROCD, and mass.

    When the input parquet contains lateral state (``raw_lat_deg``,
    ``raw_lon_deg``, ``fdm_heading_rad``), per-step lateral position is
    threaded through pyBADA via ``Lat``/``Lon``/``initialHeading``
    (loxodromic ``constantHeading=True`` integration). The commanded true
    heading per step is ``degrees(row['fdm_heading_target_rad'])`` when
    ``row['fdm_heading_target_known']`` is true, else the current heading
    is reused (no command change). Output gains ``bada_lat_deg``,
    ``bada_lon_deg``, ``bada_heading_rad`` columns.

    Args:
        flight_path: Path to the flight parquet file.
        ac: BADA aircraft object (from ``pyBADA``).
        processor: Optional flight processor with ``process_flight`` method.
        output_dir: If provided, save prediction parquet to this directory.

    Returns:
        Polars DataFrame with BADA predictions, or ``None`` on error.
    """
    if not _HAS_PYBADA:
        log.warning("pyBADA not installed — skipping BADA prediction")
        return None

    flight_path = Path(flight_path)
    flight_id = flight_path.stem

    try:
        df = pl.read_parquet(flight_path)

        if processor is not None:
            processor.process_flight(df)

        df = _backfill_cas_sel(df)

        has_lateral = (
            "fdm_heading_rad" in df.columns
            and "raw_lat_deg" in df.columns
            and "raw_lon_deg" in df.columns
        )

        results: list[dict[str, float]] = []

        current_mass = 0.85 * ac.MTOW
        current_alt = float(df["alt_std_m"][0])
        current_tas = float(df["tas_ms"][0])

        if has_lateral:
            current_lat: float | None = float(df["raw_lat_deg"][0])
            current_lon: float | None = float(df["raw_lon_deg"][0])
            current_heading_deg: float | None = float(np.degrees(df["fdm_heading_rad"][0]))
        else:
            current_lat = None
            current_lon = None
            current_heading_deg = None

        for i in range(len(df)):
            row = df.row(i, named=True)

            if i > 0 and results:
                prev = results[-1]
                current_alt = prev["Hp"] * FT
                current_tas = prev["TAS"] * _NM_TO_MS
                current_mass = prev["mass"]

            cas_ms = float(tas_to_cas(current_tas, current_alt, row["temperature"]))
            speed_type = "M" if row["mach_sel"] != 0.0 else "CAS"
            config = "CR"
            isa_temp = float(isa_temperature(current_alt))
            delta_temp = row["temperature"] - isa_temp

            rocd_target = row["vz_sel_ms"] * 60.0 / FT

            hp_init = current_alt / FT
            hp_target = row["alt_sel_m"] / FT
            phase = get_phase(hp_init, hp_target)
            ws = -row["long_wind_ms"]

            if speed_type == "M":
                v_init = row["mach"]
                v_target = row["mach_sel"]
            else:
                v_init = ms_to_kt(cas_ms)
                v_target = ms_to_kt(row["cas_sel_ms"])

            speed_diff_ratio = abs(v_init - v_target) / max(v_target, 1e-6)

            commanded_heading_deg = (
                _commanded_heading(row, current_heading_deg) if has_lateral else None
            )
            step_lat = current_lat if has_lateral else None
            step_lon = current_lon if has_lateral else None

            turn_rate_dps, bank_angle_deg, turn_direction = _step_turn_metrics(row, current_tas)

            res = _run_bada_step(
                ac=ac,
                speed_type=speed_type,
                v_init=v_init,
                v_target=v_target,
                phase=phase,
                hp_init=hp_init,
                m_init=current_mass,
                delta_temp=delta_temp,
                config=config,
                ws=ws,
                rocd_target=rocd_target,
                speed_diff_ratio=speed_diff_ratio,
                current_lat=step_lat,
                current_lon=step_lon,
                current_heading_deg=commanded_heading_deg,
                turn_rate_dps=turn_rate_dps,
                bank_angle_deg=bank_angle_deg,
                turn_direction=turn_direction,
            )

            entry: dict[str, float] = {
                "Hp": float(res["Hp"].iloc[-1]),
                "TAS": float(res["TAS"].iloc[-1]),
                "M": float(res["M"].iloc[-1]),
                "ROCD": float(res["ROCD"].iloc[-1]),
                "mass": float(res["mass"].iloc[-1]),
            }

            if has_lateral and "LAT" in res.columns and "LON" in res.columns:
                entry["LAT"] = float(res["LAT"].iloc[-1])
                entry["LON"] = float(res["LON"].iloc[-1])
                entry["HDGTrue"] = float(res["HDGTrue"].iloc[-1])
                current_lat = entry["LAT"]
                current_lon = entry["LON"]
                current_heading_deg = entry["HDGTrue"]

            results.append(entry)

        # Build output DataFrame
        df_res = pl.DataFrame(results)
        df_res = df_res.with_columns(
            [
                (pl.col("Hp") * FT).alias("bada_alt_std_m"),
                (pl.col("TAS") * _NM_TO_MS).alias("bada_tas_ms"),
                (pl.col("ROCD") * FT / 60.0).alias("bada_vz_ms"),
                pl.col("mass").alias("bada_mass_kg"),
            ]
        )
        df_res = df_res.with_columns(
            (pl.col("bada_vz_ms") / pl.col("bada_tas_ms"))
            .map_batches(lambda s: np.arcsin(s.to_numpy()))
            .alias("bada_gamma_rad")
        )

        out_cols = [
            "bada_alt_std_m",
            "bada_tas_ms",
            "bada_vz_ms",
            "bada_mass_kg",
            "bada_gamma_rad",
        ]
        if has_lateral and "LAT" in df_res.columns:
            df_res = df_res.with_columns(
                pl.col("LAT").alias("bada_lat_deg"),
                pl.col("LON").alias("bada_lon_deg"),
                (pl.col("HDGTrue") * (np.pi / 180.0)).alias("bada_heading_rad"),
            )
            out_cols.extend(["bada_lat_deg", "bada_lon_deg", "bada_heading_rad"])

        df_res = df_res.select(out_cols)

        if output_dir is not None:
            out_path = Path(output_dir) / f"{flight_id}.parquet"
            df_res.write_parquet(out_path)
            log.info("bada_prediction_saved", flight_id=flight_id, path=str(out_path))

        return df_res

    except Exception:
        log.exception("bada_prediction_failed", flight_id=flight_id)
        return None


def _run_bada_step(
    *,
    ac: Any,
    speed_type: str,
    v_init: float,
    v_target: float,
    phase: str,
    hp_init: float,
    m_init: float,
    delta_temp: float,
    config: str,
    ws: float,
    rocd_target: float,
    speed_diff_ratio: float,
    current_lat: float | None = None,
    current_lon: float | None = None,
    current_heading_deg: float | None = None,
    turn_rate_dps: float = 0.0,
    bank_angle_deg: float = 0.0,
    turn_direction: str | None = None,
) -> Any:
    """Run a single BADA TCL step, selecting the appropriate TCL function.

    When ``current_heading_deg`` is provided, the commanded heading is
    forwarded to pyBADA via ``initialHeading={'true': hdg,
    'constantHeading': True, 'magnetic': None}`` (loxodromic integration)
    along with ``Lat``/``Lon`` for lat/lon propagation.

    When ``turn_rate_dps``/``bank_angle_deg``/``turn_direction`` are non-default,
    they are forwarded to every TCL call as a ``turnMetrics`` dict shaped
    ``{'rateOfTurn', 'bankAngle', 'directionOfTurn'}`` (pyBADA 0.1.5 contract).
    """
    step_length = 4
    length = 4

    initial_heading: dict[str, Any] | None
    if current_heading_deg is not None:
        initial_heading = {
            "true": current_heading_deg,
            "constantHeading": True,
            "magnetic": None,
        }
    else:
        initial_heading = None

    lateral_kwargs: dict[str, Any] = {
        "Lat": current_lat,
        "Lon": current_lon,
        "initialHeading": initial_heading,
        "turnMetrics": {
            "rateOfTurn": float(turn_rate_dps),
            "bankAngle": float(bank_angle_deg),
            "directionOfTurn": turn_direction,
        },
    }

    if speed_diff_ratio < 0.04:
        if phase == "Cruise":
            return constantSpeedLevel(
                AC=ac,
                lengthType="time",
                length=length,
                speedType=speed_type,
                v=v_target,
                speedEvol="const",
                phase=phase,
                Hp_init=hp_init,
                m_init=m_init,
                DeltaTemp=delta_temp,
                config=config,
                step_length=step_length,
                wS=ws,
                **lateral_kwargs,
            )
        try:
            if abs(rocd_target) > 0:
                return constantSpeedROCD_time(
                    AC=ac,
                    length=length,
                    speedType=speed_type,
                    v=v_init,
                    Hp_init=hp_init,
                    ROCDtarget=rocd_target,
                    m_init=m_init,
                    DeltaTemp=delta_temp,
                    config=config,
                    step_length=step_length,
                    wS=ws,
                    **lateral_kwargs,
                )
            return constantSpeedRating_time(
                AC=ac,
                length=length,
                speedType=speed_type,
                v=v_target,
                phase=phase,
                Hp_init=hp_init,
                m_init=m_init,
                DeltaTemp=delta_temp,
                config=config,
                step_length=step_length,
                wS=ws,
                **lateral_kwargs,
            )
        except ValueError:
            return constantSpeedLevel(
                AC=ac,
                lengthType="time",
                length=length,
                speedType=speed_type,
                v=v_target,
                speedEvol="const",
                phase=phase,
                Hp_init=hp_init,
                m_init=m_init,
                DeltaTemp=delta_temp,
                config=config,
                step_length=step_length,
                wS=ws,
                **lateral_kwargs,
            )

    # Acceleration / deceleration
    speed_evol = "acc" if v_init < v_target else "dec"
    control = target(ROCDtarget=rocd_target) if rocd_target < -10 and phase != "Cruise" else None
    try:
        return accDec_time(
            AC=ac,
            length=length,
            speedType=speed_type,
            v_init=v_init,
            speedEvol=speed_evol,
            phase=phase,
            Hp_init=hp_init,
            m_init=m_init,
            DeltaTemp=delta_temp,
            config=config,
            step_length=step_length,
            wS=ws,
            control=control,
            **lateral_kwargs,
        )
    except ValueError:
        return constantSpeedLevel(
            AC=ac,
            lengthType="time",
            length=length,
            speedType=speed_type,
            v=v_target,
            speedEvol="const",
            phase=phase,
            Hp_init=hp_init,
            m_init=m_init,
            DeltaTemp=delta_temp,
            config=config,
            step_length=step_length,
            wS=ws,
            **lateral_kwargs,
        )
