#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictor for generating BADA-based flight trajectories."""

from typing import Any

import pandas as pd
import numpy as np
from pathlib import Path

from node_fdm.utils.physics.constants import ft
from pybada_predictor.utils import ms_to_kt, tas_to_cas, isa_temperature, get_phase


from pyBADA.TCL import (
    accDec_time,
    constantSpeedLevel,
    constantSpeedRating_time,
    constantSpeedROCD_time,
    target,
)


def process_single_flight(row: Any, AC: Any, processor: Any, output_dir: Path):
    """Process one flight with BADA predictions and save outputs.

    Args:
        row: Row containing flight metadata and filepath.
        AC: BADA aircraft object.
        processor: Flight processor with `process_flight`.
        output_dir: Destination directory for prediction parquet.
    """
    flight_path = Path(row.filepath)
    flight_id = flight_path.stem

    try:
        f = pd.read_parquet(flight_path)
        _ = processor.process_flight(f)

        f["cas_sel_ms"].iloc[-1] = f["cas_ms"].iloc[-1]
        f["cas_sel_ms"] = np.where(f["cas_sel_ms"] == 0.0, np.nan, f["cas_sel_ms"])
        f["cas_sel_ms"] = f["cas_sel_ms"].bfill()

        results = []
        res_vals = pd.DataFrame()

        for i, pt in f.iterrows():
            current_alt = float(pt["alt_std_m"])
            current_tas = float(pt["tas_ms"])

            if i != 0 and not res_vals.empty:
                current_alt = float(res_vals["Hp"].iloc[-1] * ft)
                current_tas = float(res_vals["TAS"].iloc[-1] * 1852 / 3600)
                current_mass = float(res_vals["mass"].iloc[-1])
            else:
                current_mass = 0.85 * AC.MTOW  # ou pt["mass_kg"]

            CAS_ms = tas_to_cas(current_tas, current_alt, pt["temperature"])
            speedType = "M" if pt["mach_sel"] != 0.0 else "CAS"
            config = "CR"
            isa_temp = isa_temperature(current_alt)
            DeltaTemp = pt["temperature"] - isa_temp

            ROCDtarget = pt["vz_sel_ms"] * 60 / ft

            Hp_init = current_alt / ft
            Hp_target = pt["alt_sel_m"] / ft
            phase = get_phase(Hp_init, Hp_target)
            m_init = current_mass
            wS = -pt["long_wind_ms"]

            if speedType == "M":
                v_init = pt["mach"]
                v_target = pt["mach_sel"]
            else:
                v_init = ms_to_kt(CAS_ms)
                v_target = ms_to_kt(pt["cas_sel_ms"])

            speed_diff_ratio = np.abs(v_init - v_target) / max(v_target, 1e-6)

            if speed_diff_ratio < 0.04:
                if phase == "Cruise":
                    res = constantSpeedLevel(
                        AC=AC,
                        lengthType="time",
                        length=4,
                        speedType=speedType,
                        v=v_target,
                        speedEvol="const",
                        phase=phase,
                        Hp_init=Hp_init,
                        m_init=m_init,
                        deltaTemp=DeltaTemp,
                        config=config,
                        step_length=4,
                        wS=wS,
                    )
                else:
                    try:
                        if np.abs(ROCDtarget) != 0:
                            res = constantSpeedROCD_time(
                                AC=AC,
                                length=4,
                                speedType=speedType,
                                v=v_init,
                                Hp_init=Hp_init,
                                ROCDtarget=ROCDtarget,
                                m_init=m_init,
                                deltaTemp=DeltaTemp,
                                config=config,
                                step_length=4,
                                wS=wS,
                            )
                        else:
                            res = constantSpeedRating_time(
                                AC=AC,
                                length=4,
                                speedType=speedType,
                                v=v_target,
                                phase=phase,
                                Hp_init=Hp_init,
                                m_init=m_init,
                                deltaTemp=DeltaTemp,
                                config=config,
                                step_length=4,
                                wS=wS,
                            )
                    except ValueError:
                        res = constantSpeedLevel(
                            AC=AC,
                            lengthType="time",
                            length=4,
                            speedType=speedType,
                            v=v_target,
                            speedEvol="const",
                            phase=phase,
                            Hp_init=Hp_init,
                            m_init=m_init,
                            deltaTemp=DeltaTemp,
                            config=config,
                            step_length=4,
                            wS=wS,
                        )
            else:
                speedEvol = "acc" if v_init < v_target else "dec"
                control = (
                    target(ROCDtarget=ROCDtarget)
                    if ROCDtarget < -10 and phase != "Cruise"
                    else None
                )
                try:
                    res = accDec_time(
                        AC=AC,
                        length=4,
                        speedType=speedType,
                        v_init=v_init,
                        speedEvol=speedEvol,
                        phase=phase,
                        Hp_init=Hp_init,
                        m_init=m_init,
                        deltaTemp=DeltaTemp,
                        config=config,
                        step_length=4,
                        wS=wS,
                        control=control,
                    )
                except ValueError:
                    res = constantSpeedLevel(
                        AC=AC,
                        lengthType="time",
                        length=4,
                        speedType=speedType,
                        v=v_target,
                        speedEvol="const",
                        phase=phase,
                        Hp_init=Hp_init,
                        m_init=m_init,
                        deltaTemp=DeltaTemp,
                        config=config,
                        step_length=4,
                        wS=wS,
                    )

            res_vals = res[["Hp", "TAS", "M", "ROCD", "mass"]].iloc[-1:]
            results.append(res_vals)

        df_res = pd.concat(results).reset_index(drop=True)
        df_res["Hp"] *= ft
        df_res["TAS"] *= 1852 / 3600
        df_res["ROCD"] *= ft / 60

        df_res = df_res.rename(
            columns={
                "TAS": "bada_tas_ms",
                "Hp": "bada_alt_std_m",
                "ROCD": "bada_vz_ms",
                "mass": "bada_mass_kg",
            }
        )
        df_res["bada_gamma_rad"] = np.arcsin(df_res.bada_vz_ms / df_res.bada_tas_ms)
        df_res = df_res.drop(columns=["M"])

        out_path = output_dir / f"{flight_id}.parquet"
        df_res.to_parquet(out_path, index=False)

    except Exception as e:
        print(f"⚠️  Erreur sur le vol {flight_id}: {e}")
