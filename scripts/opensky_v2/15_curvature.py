# %%
# %%
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

R_EARTH_M = 6371000.0


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]
process_dir = data_dir / cfg["paths"]["process_dir"]
files = os.listdir(process_dir / "A320")


def _calculate_trajectory_curvature(df, window_length=7, polyorder=2):
    if len(df) < window_length:
        return pd.Series(np.nan, index=df.index)

    time_seconds = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().values
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


def _get_all_segment_indices(df, turn_threshold):
    turn_metric = np.abs(df["track"].diff(4) / 16)
    df["in_turn"] = np.where(turn_metric >= turn_threshold, 1.0, 0.0)

    turn_mask = df["in_turn"] == 1.0
    diff_mask = np.diff(turn_mask.astype(int), prepend=0)

    boundary_indices = df.index[np.where(diff_mask != 0)[0]].tolist()

    all_indices = np.unique(np.sort([df.index[0]] + boundary_indices + [df.index[-1]]))

    return all_indices.tolist()


def _get_segment_bounds_for_pivot(all_indices, pivot_index):
    all_indices_arr = np.asarray(all_indices)

    idx_end = np.searchsorted(all_indices_arr, pivot_index, side="right")
    idx_start = idx_end - 1

    start_bound = all_indices_arr[idx_start] if idx_start >= 0 else all_indices_arr[0]

    if idx_end >= len(all_indices_arr):
        end_bound = all_indices_arr[-1]
    else:
        end_bound = all_indices_arr[idx_end]

    return start_bound, end_bound


def great_circle_distance_km(phi1, lambda1, phi2, lambda2):
    d_sigma = np.arccos(
        np.sin(phi1) * np.sin(phi2)
        + np.cos(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
    )
    return d_sigma * R_EARTH_M / 1000


def calculate_bearing(phi1, lambda1, phi2, lambda2):
    d_lambda = lambda2 - lambda1
    y = np.sin(d_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda)
    return np.arctan2(y, x)


def calculate_intermediate_point(phi1, lambda1, bearing_rad, d_rad):
    phi_C = np.arcsin(
        np.sin(phi1) * np.cos(d_rad)
        + np.cos(phi1) * np.sin(d_rad) * np.cos(bearing_rad)
    )

    y_lon = np.sin(bearing_rad) * np.sin(d_rad) * np.cos(phi1)
    x_lon = np.cos(d_rad) - np.sin(phi1) * np.sin(phi_C)
    d_lambda_AC = np.arctan2(y_lon, x_lon)

    lambda_C = lambda1 + d_lambda_AC

    return phi_C, lambda_C


def calculate_rhumb_line_track(phi1, lambda1, phi2, lambda2):

    phi1, lambda1, phi2, lambda2 = (
        np.asarray(phi1),
        np.asarray(lambda1),
        np.asarray(phi2),
        np.asarray(lambda2),
    )

    d_lambda = lambda2 - lambda1

    epsilon = 1e-12

    tan_phi2 = np.tan(phi2 / 2 + np.pi / 4)
    tan_phi1 = np.tan(phi1 / 2 + np.pi / 4)

    d_m = np.log(np.maximum(tan_phi2, epsilon) / np.maximum(tan_phi1, epsilon))

    track = np.arctan2(d_lambda, d_m)

    dm_near_zero = np.abs(d_m) < epsilon

    dm_and_dl_near_zero = dm_near_zero & (np.abs(d_lambda) < epsilon)

    track = np.where(dm_and_dl_near_zero, np.nan, track)

    dm_only_near_zero = dm_near_zero & ~dm_and_dl_near_zero

    track_ew = np.where(d_lambda > 0, np.pi / 2, -np.pi / 2)

    track = np.where(dm_only_near_zero, track_ew, track)

    return track


def process_flight_data_augmented(df_flight, turn_threshold):
    df = df_flight.copy()

    df["curvature"] = _calculate_trajectory_curvature(df)

    all_indices = _get_all_segment_indices(df, turn_threshold)

    bounds_data = [
        _get_segment_bounds_for_pivot(all_indices, index) for index in df.index
    ]

    bounds_df = pd.DataFrame(
        bounds_data, index=df.index, columns=["seg_start_idx", "seg_end_idx"]
    )
    df["seg_start_idx"] = bounds_df["seg_start_idx"]
    df["seg_end_idx"] = bounds_df["seg_end_idx"]

    lat_map = df["latitude"].to_dict()
    lon_map = df["longitude"].to_dict()
    track_map = df["track"].to_dict()
    turn_map = df["in_turn"].to_dict()

    df["lat_A"] = df["seg_start_idx"].map(lat_map)
    df["lon_A"] = df["seg_start_idx"].map(lon_map)
    df["trck_A"] = df["seg_start_idx"].map(track_map)
    df["turn_A"] = df["seg_start_idx"].map(turn_map)

    df["lat_B"] = df["seg_end_idx"].map(lat_map)
    df["lon_B"] = df["seg_end_idx"].map(lon_map)
    df["trck_B"] = df["seg_end_idx"].map(track_map)
    df["turn_B"] = df["seg_end_idx"].map(turn_map)

    phi_A, lambda_A = np.radians(df["lat_A"]), np.radians(df["lon_A"])
    phi_B, lambda_B = np.radians(df["lat_B"]), np.radians(df["lon_B"])

    # 1. Distance totale et Route Initiale
    df["seg_dist_km"] = great_circle_distance_km(phi_A, lambda_A, phi_B, lambda_B)
    df["seg_init_bearing"] = calculate_bearing(phi_A, lambda_A, phi_B, lambda_B)

    df["seg_cur_dist"] = great_circle_distance_km(
        phi_A, lambda_A, np.radians(df["latitude"]), np.radians(df["longitude"])
    ) / (R_EARTH_M / 1000)

    phi_C, lambda_C = calculate_intermediate_point(
        phi_A, lambda_A, df["seg_init_bearing"], df["seg_cur_dist"]
    )

    df["orthodromie_track"] = calculate_bearing(phi_A, lambda_A, phi_C, lambda_C)
    df["orthodromie_track"] = (df["orthodromie_track"] + 2 * np.pi) % (2 * np.pi)
    df["orthodromie_track"] = np.degrees(np.unwrap(df["orthodromie_track"]))

    df["rhumb_line_track"] = calculate_rhumb_line_track(
        phi_A, lambda_A, phi_B, lambda_B
    )
    df["rhumb_line_track"] = (df["rhumb_line_track"] + 2 * np.pi) % (2 * np.pi)
    df["rhumb_line_track"] = np.degrees(np.unwrap(df["rhumb_line_track"]))
    return df, all_indices


def track_and_groundspeed_filtered(df, window_length=9, polyorder=1):

    if len(df) < window_length:
        return df

    dt_sec = 4

    df["latitude2"] = savgol_filter(
        df["latitude"].values, window_length // 2, polyorder
    )
    df["longitude2"] = savgol_filter(
        df["longitude"].values, window_length // 2, polyorder
    )

    mean_lat_rad = np.radians(df["latitude2"].mean())

    dy_per_deg = R_EARTH_M * np.pi / 180.0
    dx_per_deg = R_EARTH_M * np.cos(mean_lat_rad) * np.pi / 180.0

    V_sol_N = (
        savgol_filter(
            df["latitude2"].values, window_length, polyorder, deriv=1, delta=dt_sec
        )
        * dy_per_deg
    )

    V_sol_E = (
        savgol_filter(
            df["longitude2"].values, window_length, polyorder, deriv=1, delta=dt_sec
        )
        * dx_per_deg
    )

    track_rad = np.arctan2(V_sol_E, V_sol_N)
    track_deg = np.degrees(track_rad)
    track_deg = np.where(track_deg > 180, track_deg - 360, track_deg)
    track_deg = np.where(track_deg < -180, track_deg + 360, track_deg)

    df["track2"] = np.degrees(np.unwrap(np.radians(track_deg)))
    return df


def _compute_heading_from_track_tas(track, u_wind, v_wind, tas, gs=None):

    MS_TO_KTS = 1.94384

    u_wind = u_wind * MS_TO_KTS
    v_wind = v_wind * MS_TO_KTS

    ws = np.sqrt(u_wind**2 + v_wind**2)

    wind_direction_rad = np.arctan2(u_wind, v_wind)
    wind_direction_deg = np.degrees(wind_direction_rad)

    wind_direction_from = (wind_direction_deg + 180) % 360

    angle_track_minus_wd = np.radians(wind_direction_from - track)

    sin_wca = ws * np.sin(angle_track_minus_wd) / tas

    if np.isscalar(track):
        track = np.array([track])
        sin_wca = np.array([sin_wca])

    heading = np.full_like(track, np.nan, dtype=float)

    valid_indices = np.abs(sin_wca) <= 1

    if np.any(valid_indices):
        wca_deg = np.degrees(np.arcsin(sin_wca[valid_indices]))
        heading[valid_indices] = (track[valid_indices] + wca_deg) % 360

    return heading.item() if heading.size == 1 else heading


def clean_and_interpolate_track(df_flight, n_diff=2):
    df = df_flight.copy()
    track_diff_metric = np.abs(df["track"].diff(n_diff) / (n_diff * 4))

    df["track"] = np.where(track_diff_metric > 5, np.nan, df["track"])

    df["track"] = df["track"].interpolate(method="linear")

    return df


def process_flight_for_heading(df_flight):
    df = df_flight.copy()

    df["heading"] = df.apply(
        lambda row: _compute_heading_from_track_tas(
            row["track"],
            row["u_component_of_wind"],
            row["v_component_of_wind"],
            row["TAS"],
        ),
        axis=1,
    )

    df["heading"] = np.degrees(np.unwrap(np.radians(df["heading"])))

    return df


cut = 1000

for i in range(0, 2):
    f = pd.read_parquet(process_dir / "A320" / files[i])
    f["track"] = np.degrees(np.unwrap(np.radians(f.track)))
    f = clean_and_interpolate_track(f, n_diff=2)
    f = process_flight_for_heading(f)
    f["turning_rate"] = f["heading"].diff(2) / (2 * 4.0)
    f, all_indices = process_flight_data_augmented(f, turn_threshold=0.15)
    plt.scatter(f.longitude, f.latitude, c=f["in_turn"], cmap="coolwarm", s=1)

    segments2 = [el for el in all_indices]  # if (el >= cut - 200) and (el < cut + 200)]
    plt.scatter(f.longitude.loc[segments2], f.latitude.loc[segments2], c="g", s=5)
    plt.show()

    plt.plot(f.orthodromie_track % 360, color="b", lw=3)
    plt.plot(f.rhumb_line_track, color="orange")
    plt.plot(f.track, color="g")
    plt.show()
    # %%


# %%
plt.plot(f.seg_cur_dist, color="g")


# %%
