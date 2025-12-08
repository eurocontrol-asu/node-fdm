import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


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
    d_track = np.where(d_track > 180, d_track - 360, d_track)
    d_track = np.where(d_track < -180, d_track + 360, d_track)

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


def find_segment_bounds(all_indices, pivot_index, total_length):
    all_indices = np.asarray(all_indices)

    idx_end = np.searchsorted(all_indices, pivot_index, side="right")
    idx_start = idx_end - 1

    if idx_start < 0:
        start_bound = 0
    else:
        start_bound = all_indices[idx_start]

    if idx_end >= len(all_indices):
        end_bound = total_length - 1
    else:
        end_bound = all_indices[idx_end]

    return start_bound, end_bound


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
    track_deg = (np.degrees(theta_rad) + 360) % 360

    return pd.Series(track_deg, index=df.index)


def add_theoretical_track_to_dataframe(df):
    track_simulated = np.rad2deg(np.unwrap(np.deg2rad(df["track"].values)))
    time_data = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().values

    turning_indices = detect_start_of_turning_points(
        track_simulated,
        time_data,
        threshold_deg_per_sec=0.05,
        noise_threshold_deg_per_sec=0.005,
    )
    df = augment_dataframe_with_segment_coords(df, turning_indices)
    df["track_sel"] = calculate_theoretical_track(df)
    return df
