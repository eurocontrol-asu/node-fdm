# %%
import os
from turtle import color
import yaml
from pathlib import Path
import pandas as pd

import numpy as np
from scipy.signal import savgol_filter, find_peaks

import matplotlib.pyplot as plt


def detect_start_of_turning_points(
    track_degrees,
    time_seconds,
    threshold_deg_per_sec=0.05,
    noise_threshold_deg_per_sec=0.005,
):
    window_length = 9
    polyorder = 3
    if len(track_degrees) < window_length:
        return np.array([])

    smoothed_track = savgol_filter(track_degrees, window_length, polyorder)

    d_track = np.diff(smoothed_track, prepend=smoothed_track[0])
    dt = np.diff(time_seconds, prepend=1)

    dt = np.where(dt < 1e-6, 1e-6, dt)
    rotation_rate = d_track / dt

    abs_rotation_rate = np.abs(rotation_rate)

    peaks, _ = find_peaks(abs_rotation_rate, height=threshold_deg_per_sec, distance=10)

    start_indices = []

    for peak_index in peaks:
        current_index = peak_index

        while current_index > 0:
            current_index -= 1

            if abs_rotation_rate[current_index] < noise_threshold_deg_per_sec:
                start_indices.append(current_index + 1)
                break

            if current_index == 0:
                start_indices.append(0)
                break

    return np.unique(np.array(start_indices))

tr


def augment_dataframe_with_segment_coords(df, turning_indices):
    if df.empty:
        return df

    total_length = len(df)

    bounds_series = df.index.to_series().apply(
        lambda i: find_segment_bounds(turning_indices, i, total_length)
    )

    df[["segment_start_idx", "segment_end_idx"]] = bounds_series.apply(pd.Series)

    lat_map = df["latitude"].to_dict()
    lon_map = df["longitude"].to_dict()

    df["lat_A"] = df["segment_start_idx"].map(lat_map)
    df["lon_A"] = df["segment_start_idx"].map(lon_map)

    df["lat_B"] = df["segment_end_idx"].map(lat_map)
    df["lon_B"] = df["segment_end_idx"].map(lon_map)
    return df


def calculate_theoretical_track(df):
    phi_C = np.radians(df["latitude"])
    lambda_C = np.radians(df["longitude"])

    phi_B = np.radians(df["lat_B"])
    lambda_B = np.radians(df["lon_B"])

    d_lambda_CB = lambda_B - lambda_C

    y = np.sin(d_lambda_CB) * np.cos(phi_B)

    x = np.cos(phi_C) * np.sin(phi_B) - np.sin(phi_C) * np.cos(phi_B) * np.cos(
        d_lambda_CB
    )

    theta_rad = np.arctan2(y, x)
    track_deg = np.degrees(theta_rad)

    return pd.Series(track_deg, index=df.index)


def add_theoretical_track_to_dataframe(df):
    time_data = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().values

    turning_indices = detect_start_of_turning_points(
        df["track"],
        time_data,
        threshold_deg_per_sec=0.05,
        noise_threshold_deg_per_sec=0.005,
    )
    df = augment_dataframe_with_segment_coords(df, turning_indices)
    df["track_sel"] = calculate_theoretical_track(df)
    return df, turning_indices


import numpy as np


import numpy as np
from scipy.signal import savgol_filter
import pandas as pd


import numpy as np
from scipy.signal import savgol_filter
import pandas as pd

# Rayon terrestre moyen en mètres (utilisé pour la conversion Lat/Lon -> mètres)
R_EARTH_M = 6371000.0


def track_and_groundspeed_filtered(
    df, time_col="timestamp", window_length=9, polyorder=1
):

    if len(df) < window_length:
        print("Erreur: Données trop courtes pour le lissage.")
        return df

    dt_sec = 4

    # --- LISSAGE DES COORDONNÉES ---

    # Appliquer le filtre Savitzky-Golay à la Latitude et à la Longitude
    df["latitude2"] = savgol_filter(
        df["latitude"].values, window_length // 2, polyorder
    )
    df["longitude2"] = savgol_filter(
        df["longitude"].values, window_length // 2, polyorder
    )

    # 2. CALCUL DES COMPOSANTES DE VITESSE (Utilisation de S-G avec deriv=1)

    # Latitude moyenne du segment pour l'approximation locale
    mean_lat_rad = np.radians(df["latitude2"].mean())

    # Facteurs de conversion (de degré/seconde à mètre/seconde)
    dy_per_deg = R_EARTH_M * np.pi / 180.0  # Vitesse Nord (m/s)
    dx_per_deg = R_EARTH_M * np.cos(mean_lat_rad) * np.pi / 180.0  # Vitesse Est (m/s)

    # Vitesse Nord (Vy ou VN) = Dérivée de la Latitude
    V_sol_N = (
        savgol_filter(
            df["latitude2"].values, window_length, polyorder, deriv=1, delta=dt_sec
        )
        * dy_per_deg
    )

    # Vitesse Est (Vx ou VE) = Dérivée de la Longitude
    V_sol_E = (
        savgol_filter(
            df["longitude2"].values, window_length, polyorder, deriv=1, delta=dt_sec
        )
        * dx_per_deg
    )

    # 3. CALCUL DE LA VITESSE SOL (Ground Speed)

    # V_sol (magnitude) = sqrt(V_sol_E² + V_sol_N²)
    V_sol = np.sqrt(V_sol_E**2 + V_sol_N**2)

    # 4. CALCUL DE LA ROUTE SOL (Track)

    # Track = Azimut du vecteur vitesse (Est/Nord)
    # np.arctan2(y, x) -> y=Vx, x=Vy pour obtenir l'angle par rapport au Nord
    track_rad = np.arctan2(V_sol_E, V_sol_N)
    track_deg = np.degrees(track_rad)
    track_deg = np.where(track_deg > 180, track_deg - 360, track_deg)
    track_deg = np.where(track_deg < -180, track_deg + 360, track_deg)

    df["track2"] = np.degrees(np.unwrap(np.radians(track_deg)))
    return df


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]
process_dir = data_dir / cfg["paths"]["process_dir"]
files = os.listdir(process_dir / "A320")

for i in range(6, 7):
    f = pd.read_parquet(process_dir / "A320" / files[i])

    f = track_and_groundspeed_filtered(f)
    f, turns = add_theoretical_track_to_dataframe(f)
    # plt.plot(f.track_sel, color="r")
    plt.plot(np.degrees(np.unwrap(np.radians(f.track))), color="b")
    plt.plot(np.degrees(np.unwrap(np.radians(f.track2))), color="r")
    plt.show()
# %%
n = 1000
plt.plot(f.longitude.iloc[n:], f.latitude.iloc[n:], color="r")
plt.scatter(f.lon_A.iloc[n:], f.lat_A.iloc[n:], color="b")

# %%
plt.plot(f.track2.diff(3).iloc[n:], lw=1, color="r")
turn2 = [el for el in turns if el >= n]
plt.scatter(turn2, f.track2.diff(3).iloc[turn2], color="b")
# %%
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd

R_EARTH_M = 6371000.0


def calculate_trajectory_curvature(
    df, window_length=7, polyorder=2, time_col="timestamp"
):
    if len(df) < window_length:
        return pd.Series(np.nan, index=df.index)

    time_seconds = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values
    dt_sec = np.mean(np.diff(time_seconds))
    dt_sec = np.maximum(dt_sec, 1e-6)

    mean_lat_rad = np.radians(df["latitude"].mean())

    dy_per_deg = R_EARTH_M * np.pi / 180.0
    dx_per_deg = R_EARTH_M * np.cos(mean_lat_rad) * np.pi / 180.0

    y_coords_m = (df["latitude"].values - df["latitude"].iloc[0]) * dy_per_deg
    x_coords_m = (df["longitude"].values - df["longitude"].iloc[0]) * dx_per_deg

    x_dot = savgol_filter(x_coords_m, window_length, polyorder, deriv=1, delta=dt_sec)
    y_dot = savgol_filter(y_coords_m, window_length, polyorder, deriv=1, delta=dt_sec)

    x_ddot = savgol_filter(x_coords_m, window_length, polyorder, deriv=2, delta=dt_sec)
    y_ddot = savgol_filter(y_coords_m, window_length, polyorder, deriv=2, delta=dt_sec)

    numerator = x_dot * y_ddot - y_dot * x_ddot

    denominator_sq = x_dot**2 + y_dot**2
    denominator_sq[denominator_sq < 1e-6] = 1e-6

    denominator = denominator_sq**1.5

    kappa_signed = numerator / denominator

    return pd.Series(kappa_signed, index=df.index)


def find_inflection_points(kappa_signed):
    sign_array = np.sign(kappa_signed)
    sign_change = np.diff(sign_array)
    inflection_indices = np.where(sign_change != 0)[0] + 1
    return inflection_indices


def detect_maneuver_bounds_kappa(
    df, kappa_col="kappa", high_threshold_kappa=0.00001, min_segment_length=3
):

    kappa_signed = df[kappa_col].values

    inflection_indices = find_inflection_points(kappa_signed)

    abs_kappa = np.abs(kappa_signed)
    maneuver_mask = abs_kappa > high_threshold_kappa

    diff_mask = np.diff(maneuver_mask.astype(int), prepend=0)

    start_indices = np.where(diff_mask == 1)[0]
    end_indices = np.where(diff_mask == -1)[0] - 1

    if maneuver_mask[-1]:
        end_indices = np.append(end_indices, len(df) - 1)

    all_transition_indices = np.sort(
        np.unique(np.concatenate([start_indices, end_indices, inflection_indices]))
    )

    final_indices = [0]
    for i in range(len(all_transition_indices) - 1):
        idx_start = all_transition_indices[i]
        idx_end = all_transition_indices[i + 1]

        if idx_end - idx_start >= min_segment_length:
            final_indices.append(idx_end)

    return np.unique(np.array(final_indices))


for i in range(7, 15):
    f = pd.read_parquet(process_dir / "A320" / files[i])

    f["curv"] = calculate_trajectory_curvature(f)
    high_threshold_kappa = 0.3
    coeff = np.where(
        np.abs((f.track.diff(4) / 16)) > high_threshold_kappa, f.curv, np.nan
    )
    f["curv_keep"] = coeff
    coeff = np.where(
        np.abs((f.track.diff(4) / 16)) >= high_threshold_kappa, 1.0, np.nan
    )
    f["curv_keep2"] = coeff
    coeff2 = np.where(
        np.abs((f.track.diff(4) / 16)) >= high_threshold_kappa, np.nan, 1.0
    )
    f["curv_keep3"] = coeff2
    condition = f["curv_keep"].diff(1).isna() ^ f["curv_keep"].diff(-1).isna()
    indices_true = f.index[condition]
    indices_true
    n = len(f)
    delta = 200
    for n in np.arange(0, n, delta):
        turn2 = [el for el in indices_true if (el >= n) and (el < n + delta)]
        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        plt.plot(
            (f.longitude * f["curv_keep2"]).iloc[n : n + delta],
            (f.latitude * f["curv_keep2"]).iloc[n : n + delta],
            color="r",
            transform=ccrs.PlateCarree(),
        )

        plt.plot(
            (f.longitude * f["curv_keep3"]).iloc[n : n + delta],
            (f.latitude * f["curv_keep3"]).iloc[n : n + delta],
            color="g",
            transform=ccrs.PlateCarree(),
        )
        plt.scatter(
            f.longitude.iloc[turn2],
            f.latitude.iloc[turn2],
            color="b",
            transform=ccrs.PlateCarree(),
        )
        plt.show()
# %%
f["curv_keep"] * f.longitude
# %%
from traffic.core.flight import Flight

t = Flight.from_file(process_dir / "A320" / files[6])
f2 = t.simplify(1000).data
plt.plot(f.longitude, f.latitude)
plt.scatter(f2.longitude, f2.latitude)
plt.plot(
    (f.longitude * f["curv_keep2"]).iloc[n:],
    (f.latitude ** f["curv_keep2"]).iloc[n:],
    color="r",
)

plt.plot(
    (f.longitude * f["curv_keep3"]).iloc[n:],
    (f.latitude ** f["curv_keep3"]).iloc[n:],
    color="g",
)
plt.xlim(-0.3, 0.3)
plt.ylim(50.6, 51.5)
# %%

plt.plot(
    np.arange(n, len(f)),
    np.where(np.abs(f.curv.iloc[n:]) > high_threshold_kappa, f.curv.iloc[n:], np.nan),
    color="r",
)
plt.scatter(turn2, f.curv.iloc[turn2], color="b")
# %%
plt.plot(f.altitude.iloc[n:], color="b")

# %%
plt.plot(f.latitude.diff(4).iloc[n:], color="b")
# %%
plt.plot((f.track2.diff(4) / 16).iloc[n:], color="r")
plt.scatter(turn2, (f.track2.diff(4) / 16).iloc[turn2], color="b")
# %%
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd


def calculate_track_via_local_velocity(df):

    # 1. PARAMÈTRES DU FILTRE (Doivent être identiques pour le lissage des coordonnées)
    window_length = 9
    polyorder = 3
    if len(df) < window_length:
        print("Erreur: Données trop courtes.")
        return df

    R_earth_m = 6371000.0

    dt_sec = 4
    mean_lat_rad = np.radians(df["latitude"].mean())

    dy_per_deg = R_earth_m * np.pi / 180.0
    dx_per_deg = R_earth_m * np.cos(mean_lat_rad) * np.pi / 180.0

    lon_smooth = savgol_filter(f["longitude"].values, 7, 3)
    lat_smooth = savgol_filter(f["latitude"].values, 7, 3)
    Vy = (
        savgol_filter(lat_smooth, window_length, polyorder, deriv=1, delta=dt_sec)
        * dy_per_deg
    )

    Vx = (
        savgol_filter(lon_smooth, window_length, polyorder, deriv=1, delta=dt_sec)
        * dx_per_deg
    )
    track_rad = np.arctan2(Vx, Vy)
    track_deg = np.degrees(track_rad)
    df["track_velocity"] = track_deg
    df["track_velocity"] = (df["track_velocity"] + 360) % 360
    return df


for i in range(7, 8):
    f = pd.read_parquet(process_dir / "A320" / files[i])
    f = calculate_track_via_local_velocity(f)

    plt.plot(f.track)
    plt.plot(f.track_velocity, color="g")
    plt.show()

# %%
from node_fdm.utils.physics.constants import ftmn, kt

import sys

sys.path.append("/data/common/dataiku/config/projects/NODEFDM/lib/python/src/node_fdm")


for i in range(10):
    f = pd.read_parquet(process_dir / "A320" / files[i])

    f["gamma"] = np.arcsin((f["vertical_rate"] * ftmn) / (f["groundspeed"] * kt))
    gamma_segments, _ = detect_constant_segments(f, var_name="gamma", **gamma_cfg)
    f = add_segment_column(f, gamma_segments, "gamma_sel", fill_value=np.nan)

    plt.plot(f.gamma)
    plt.plot(f.gamma_sel)
    plt.show()


# %%


gamma_cfg = {
    "tol": 0.002,
    "min_len": 15,
    "use_alt": False,
    # "min_abs_value": 0.002,
    "smooth_window": 5,
    "smooth_method": "savgol",
}


vz_cfg = {
    "tol": 25,
    "min_len": 25,
    "use_alt": False,
    "min_abs_value": 75,
    "smooth_window": 15,
    "smooth_method": "savgol",
}

for i in range(0, 16):
    f = pd.read_parquet(process_dir / "A320" / files[i])

    f["gamma"] = np.arcsin((f["vertical_rate"] * ftmn) / (f["groundspeed"] * kt))

    vz_segments, _ = detect_constant_segments(f, var_name="vertical_rate", **vz_cfg)
    f = add_segment_column(f, vz_segments, "vz_sel", fill_value=np.nan)

    f_gamma = f.copy()
    t = f["time"] if "time" in f.columns else f.index
    if len(vz_segments) > 0:
        for seg in vz_segments:
            mask = (t >= seg["start_time"]) & (t <= seg["end_time"])
            f_gamma.loc[mask, "gamma"] = np.nan

    gamma_segments, _ = detect_constant_segments(f_gamma, var_name="gamma", **gamma_cfg)
    f = add_segment_column(f, gamma_segments, "gamma_sel", fill_value=np.nan)

    plt.plot(f.vertical_rate)
    plt.plot(f.vz_sel, color="r")
    plt.show()

    plt.plot(f.gamma)
    plt.plot(f.gamma_sel)
    plt.show()

# %%


def build_spd_and_vert_selected_from_segments(f: pd.DataFrame, config) -> pd.DataFrame:
    """Build selected variables (Mach, CAS, vertical_rate, altitude) from detected segments.

    Args:
        f: Input flight DataFrame.
        config: Segment detection configuration dictionary.

    Returns:
        DataFrame with selected variables added.
    """
    f = f.copy()

    mach_cfg = config.get("mach", {})
    mach_segments, _ = detect_constant_segments(f, var_name="Mach", **mach_cfg)
    f = add_segment_column(f, mach_segments, "mach_sel", fill_value=np.nan)

    f_cas = f.copy()
    t = f["time"] if "time" in f.columns else f.index
    if len(mach_segments) > 0:
        for seg in mach_segments:
            mask = (t >= seg["start_time"]) & (t <= seg["end_time"])
            f_cas.loc[mask, "CAS"] = np.nan

    cas_cfg = config.get("cas", {})
    cas_segments, _ = detect_constant_segments(f_cas, var_name="CAS", **cas_cfg)
    f = add_segment_column(f, cas_segments, "cas_sel", fill_value=np.nan)

    vz_cfg = config.get("vz", {})
    vz_segments, _ = detect_constant_segments(f, var_name="vertical_rate", **vz_cfg)
    f = add_segment_column(f, vz_segments, "vz_sel", fill_value=np.nan)

    gamma_cfg = config.get("gamma", None)
    if gamma_cfg is not None:
        f_gamma = f.copy()
        t = f["time"] if "time" in f.columns else f.index
        if len(vz_segments) > 0:
            for seg in vz_segments:
                mask = (t >= seg["start_time"]) & (t <= seg["end_time"])
                f_gamma.loc[mask, "gamma"] = np.nan
        gamma_segments, _ = detect_constant_segments(
            f_gamma, var_name="gamma", **gamma_cfg
        )
        f = add_segment_column(f, gamma_segments, "gamma_sel", fill_value=np.nan)

    alt_cfg = config.get("alt", None)
    if alt_cfg is not None:
        alt_segments, _ = detect_constant_segments(f, var_name="altitude", **alt_cfg)
        f = add_segment_column(f, alt_segments, "selected_mcp", fill_value=np.nan)
        f.loc[f.index[-1], "selected_mcp"] = f.loc[f.index[-1], "altitude"]
        f["selected_mcp"] = f["selected_mcp"].bfill()

    return f


# %%

selected_param_config = {
    "mach": {
        "tol": 0.0005,
        "min_len": 120,
        "alt_threshold": 15000,
        "smooth_window": 30,
        "use_alt": True,
    },
    "cas": {
        "tol": 0.75,
        "min_len": 20,
        "use_alt": False,
        "smooth_window": 20,
        "smooth_method": "savgol",
    },
    "vz": {
        "tol": 25,
        "min_len": 25,
        "use_alt": False,
        "min_abs_value": 75,
        "smooth_window": 15,
        "smooth_method": "savgol",
    },
    "alt": {
        "tol": 25,
        "min_len": 5,
        "use_alt": False,
        "min_abs_value": 25,
        "smooth_window": 5,
        "smooth_method": "savgol",
    },
    "gamma": {
        "tol": 0.002,
        "min_len": 15,
        "use_alt": False,
        "smooth_window": 5,
        "smooth_method": "savgol",
    },
}

for i in range(0, 16):
    f = pd.read_parquet(process_dir / "A320" / files[i])

    f["gamma"] = np.arcsin((f["vertical_rate"] * ftmn) / (f["groundspeed"] * kt))
    f = build_spd_and_vert_selected_from_segments(f, selected_param_config)
    cols = ["Mach", "CAS", "vertical_rate", "gamma"]
    sel_cols = ["mach_sel", "cas_sel", "vz_sel", "gamma_sel"]
    for col, sel_col in zip(cols, sel_cols):
        plt.plot(f[col], label=col)
        plt.plot(f[sel_col], label=f"{col}_sel")
        plt.legend()
        plt.show()
        break
    # %%
