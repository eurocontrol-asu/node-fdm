# %%

import pandas as pd
import numpy as np
from pathlib import Path

from pyBADA.TCL import (
    accDec,
    constantSpeedRating,
)

import sys
import yaml

sys.path.append(str(Path.cwd().parents[0]))
from pyBADA.bada4 import Bada4Aircraft

import matplotlib.pyplot as plt

cfg = yaml.safe_load(open("config.yaml"))
data_dir = Path(cfg["paths"]["data_dir"])
bada_4_2_dir = (
    data_dir / cfg["paths"]["bada_4_2_dir"]
)  # Replace by yout BADA 4.2 directory

acft = "A320-214"
AC = Bada4Aircraft("4.2", filePath=bada_4_2_dir, acName=acft)


from pyBADA.bada3 import Bada3Aircraft

bada_316_dir = "/data/common/dataiku/config/projects/FUEL_MODEL/lib/python/BADA/3.16"
AC = Bada3Aircraft("3.17", filePath=bada_316_dir, acName="A320")


# Constantes
GAMMA = 1.4
R = 287.05
KT_TO_MS = 0.514444  # 1 noeud = 0.514444 m/s


# ISA Pressure and Temperature at a given altitude (in m)
def isa_pressure(h):
    T0 = 288.15
    p0 = 101325
    if h <= 11000:
        T = T0 - 0.0065 * h
        p = p0 * (T / T0) ** 5.2561
    else:
        T = 216.65
        p = 22632 * np.exp(-9.80665 * (h - 11000) / (R * T))
    return p, T


# Mach & altitude -> CAS (Renvoie des m/s)
def cas_from_mach_alt(M, h):
    p, T = isa_pressure(h)

    qc = p * ((1 + 0.2 * M * M) ** 3.5 - 1)

    M_cas = np.sqrt(5 * (((qc / 101325 + 1) ** (2 / 7) - 1)))
    CAS_ms = 340.294 * M_cas
    return CAS_ms


def crossover_alt(CAS_target_kt, Mach_target):
    CAS_target_ms = CAS_target_kt * KT_TO_MS

    for h_ft in range(0, 45000, 10):
        h_m = h_ft * 0.3048  # ft -> m
        CAS_at_M_ms = cas_from_mach_alt(Mach_target, h_m)

        if CAS_at_M_ms <= CAS_target_ms:
            return h_ft

    return None


def update_time_conso(df, total_time, total_conso, total_dist):
    df["time"] += total_time
    total_time = df["time"].values[-1]
    df["FUELCONSUMED"] += total_conso
    total_conso = df["FUELCONSUMED"].values[-1]
    df["dist"] += total_dist
    total_dist = df["dist"].values[-1]
    return df, total_time, total_conso, total_dist


def simulate_descent(
    AC,
    h_init,
    h_final,
    cas_target,
    mach_target,
    level_segments,  # liste de tuples : (altitude_ft, speed_target CAS)
    mass_init,
    deltaTemp=0,
    wind=0.0,
    h_step=500,
):
    total_time = 0.0
    total_conso = 0.0
    total_dist = 0.0
    df_total = []
    mass = mass_init

    # ---------- 1. Altitude crossover ----------
    h_cross = crossover_alt(cas_target, mach_target)
    print(f">>> Crossover altitude = {h_cross} ft")

    # ---------- 2. Segment Mach constant ----------
    df_M = constantSpeedRating(
        AC,
        "M",
        mach_target,
        h_init,
        h_cross,
        mass,
        deltaTemp,
        wS=wind,
        Hp_step=h_step,
    )
    df_M, total_time, total_conso, total_dist = update_time_conso(
        df_M, total_time, total_conso, total_dist
    )
    df_total.append(df_M)
    mass = df_M["mass"].values[-1]

    if len(level_segments) == 0:
        h_target = h_final
    else:
        h_target = level_segments[0][0]

    # ---------- 3. Segment CAS constant ----------
    df_CAS = constantSpeedRating(
        AC,
        "CAS",
        cas_target,
        h_cross,
        h_target,
        mass,
        deltaTemp,
        wS=wind,
        Hp_step=h_step,
    )
    df_CAS, total_time, total_conso, total_dist = update_time_conso(
        df_CAS, total_time, total_conso, total_dist
    )
    df_total.append(df_CAS.iloc[1:])
    mass = df_CAS["mass"].values[-1]
    last_alt = df_CAS["Hp"].values[-1]
    last_speed = cas_target
    for i, (seg_alt, seg_speed) in enumerate(level_segments):

        if i != 0:
            df_desc = constantSpeedRating(
                AC,
                "CAS",
                last_speed,
                last_alt,
                seg_alt,
                mass,
                deltaTemp,
                wS=wind,
                Hp_step=h_step,
            )
            df_desc, total_time, total_conso, total_dist = update_time_conso(
                df_desc, total_time, total_conso, total_dist
            )
            df_total.append(df_desc)
            mass = df_desc["mass"].values[-1]

        df_acc = accDec(
            AC,
            "CAS",
            last_speed,
            seg_speed,
            "Cruise",
            seg_alt,
            mass,
            deltaTemp,
            wS=wind,
            config="CR",
        )
        df_acc, total_time, total_conso, total_dist = update_time_conso(
            df_acc, total_time, total_conso, total_dist
        )
        df_total.append(df_acc)
        mass = df_acc["mass"].values[-1]
        last_alt = seg_alt
        last_speed = seg_speed

    if len(level_segments) >= 0:
        df_final = constantSpeedRating(
            AC,
            "CAS",
            last_speed,
            last_alt,
            h_final,
            mass,
            deltaTemp,
            wS=wind,
            Hp_step=h_step,
        )
        df_final, total_time, total_conso, total_dist = update_time_conso(
            df_final, total_time, total_conso, total_dist
        )
        df_total.append(df_final)
    df = pd.concat(df_total, ignore_index=True)
    return df


# %%
df = simulate_descent(
    AC,
    h_init=35000,
    h_final=3000,
    cas_target=270,
    mach_target=0.78,
    level_segments=[(10000, 250), (6000, 220)],
    mass_init=64000,
    deltaTemp=0,
    wind=0.0,
)

# %%

fig, axs = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle("Profil de Vol – BADA4", fontsize=18, fontweight="bold")

# ============================
# 1. Altitude (ft)
# ============================
ax = axs[0, 0]
ax.plot(df["time"], df["Hp"], color="navy")
ax.set_ylabel("Altitude (ft)")
ax.set_title("Altitude vs Temps")
ax.grid(True)

# ============================
# 2. Distance (NM)
# ============================
ax = axs[0, 1]
ax.plot(df["time"], df["dist"], color="darkgreen")
ax.set_ylabel("Distance (NM)")
ax.set_title("Distance parcourue vs Temps")
ax.grid(True)

# ============================
# 3. CAS (kt)
# ============================
ax = axs[1, 0]
ax.plot(df["time"], df["CAS"], color="firebrick")
ax.set_ylabel("CAS (kt)")
ax.set_title("CAS vs Temps")
ax.grid(True)

# ============================
# 4. Mach
# ============================
ax = axs[1, 1]
ax.plot(df["time"], df["M"], color="purple")
ax.set_ylabel("Mach")
ax.set_title("Mach vs Temps")
ax.grid(True)

# ============================
# 5. Masse (kg)
# ============================
ax = axs[2, 0]
ax.plot(df["time"], df["mass"], color="darkorange")
ax.set_ylabel("Masse (kg)")
ax.set_xlabel("Temps (s)")
ax.set_title("Masse vs Temps")
ax.grid(True)

# ============================
# 6. Fuel consommé (kg)
# ============================
ax = axs[2, 1]
ax.plot(df["time"], df["FUELCONSUMED"], color="black")
ax.set_ylabel("Fuel (kg)")
ax.set_xlabel("Temps (s)")
ax.set_title("Fuel consommé vs Temps")
ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# %%


# %%

acft = "A320-214"
AC = Bada4Aircraft("4.2", filePath=bada_4_2_dir, acName=acft)
constantSpeedRating(AC,
"M",
0.76,
35000.0,
30060.0,
64000.0,
0.0, wS=0.0, h_step= 500.0)
# %%
print(AC)
# %%
bada_316_dir = "/data/common/dataiku/config/projects/FUEL_MODEL/lib/python/BADA/3.16"
AC = Bada3Aircraft("3.16", filePath=bada_316_dir, acName="A320")
print(AC)
# %%
bada_4_2_dir
# %%
