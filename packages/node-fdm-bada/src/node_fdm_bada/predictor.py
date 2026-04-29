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

        # Back-fill selected CAS (replace zeros with NaN first)
        cas_sel = df["cas_sel_ms"].to_numpy().copy()
        cas_sel[-1] = df["cas_ms"][-1]
        cas_sel[cas_sel == 0.0] = np.nan
        # Back-fill NaN values
        for i in range(len(cas_sel) - 2, -1, -1):
            if np.isnan(cas_sel[i]):
                cas_sel[i] = cas_sel[i + 1]
        df = df.with_columns(pl.Series("cas_sel_ms", cas_sel))

        results: list[dict[str, float]] = []

        current_mass = 0.85 * ac.MTOW
        current_alt = float(df["alt_std_m"][0])
        current_tas = float(df["tas_ms"][0])

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
            )

            results.append(
                {
                    "Hp": float(res["Hp"].iloc[-1]),
                    "TAS": float(res["TAS"].iloc[-1]),
                    "M": float(res["M"].iloc[-1]),
                    "ROCD": float(res["ROCD"].iloc[-1]),
                    "mass": float(res["mass"].iloc[-1]),
                }
            )

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
        df_res = df_res.select(
            [
                "bada_alt_std_m",
                "bada_tas_ms",
                "bada_vz_ms",
                "bada_mass_kg",
                "bada_gamma_rad",
            ]
        )

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
) -> Any:
    """Run a single BADA TCL step, selecting the appropriate TCL function."""
    step_length = 4
    length = 4

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
                deltaTemp=delta_temp,
                config=config,
                step_length=step_length,
                wS=ws,
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
                    deltaTemp=delta_temp,
                    config=config,
                    step_length=step_length,
                    wS=ws,
                )
            return constantSpeedRating_time(
                AC=ac,
                length=length,
                speedType=speed_type,
                v=v_target,
                phase=phase,
                Hp_init=hp_init,
                m_init=m_init,
                deltaTemp=delta_temp,
                config=config,
                step_length=step_length,
                wS=ws,
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
                deltaTemp=delta_temp,
                config=config,
                step_length=step_length,
                wS=ws,
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
            deltaTemp=delta_temp,
            config=config,
            step_length=step_length,
            wS=ws,
            control=control,
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
            deltaTemp=delta_temp,
            config=config,
            step_length=step_length,
            wS=ws,
        )
