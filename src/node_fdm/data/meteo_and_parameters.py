#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Meteorological preprocessing utilities and parameter derivations."""

from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


from joblib import Parallel, delayed

from node_fdm.utils.physics.constants import T0, p0, g, R, gamma_ratio, a0, ftmn, kt


def detect_constant_segments(
    f: pd.DataFrame,
    var_name: str,
    tol: float = 0.002,
    min_len: int = 5,
    alt_threshold: float = 20000,
    use_alt: bool = True,
    min_abs_value: Any = None,
    smooth_window: Any = None,
    smooth_method: str = "rolling",
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Detect segments with quasi-constant values for a given variable.

    Args:
        f: Input DataFrame containing the variable.
        var_name: Column name to analyze for constant segments.
        tol: Difference tolerance to consider a point stable.
        min_len: Minimum length for a segment to be kept.
        alt_threshold: Minimum altitude to consider when `use_alt` is True.
        use_alt: Whether to use altitude constraint when detecting segments.
        min_abs_value: Optional minimum absolute value for segment inclusion.
        smooth_window: Optional window length for smoothing.
        smooth_method: Smoothing method (`rolling` or `savgol`).

    Returns:
        Tuple of (segments list, smoothed series values).
    """
    if var_name not in f.columns:
        raise ValueError(f"Variable '{var_name}' not found in DataFrame")

    if "time" in f.columns:
        time = f["time"].values
    else:
        time = f.index.values

    y = f[var_name].astype(float).values

    if smooth_window is not None and smooth_window > 1:
        if smooth_method == "rolling":
            y = (
                pd.Series(y)
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .values
            )
        elif smooth_method == "savgol":
            win = min(smooth_window, len(y) - (len(y) % 2 == 0))
            y = savgol_filter(y, window_length=win, polyorder=2, mode="interp")

    alt = None
    if use_alt:
        if "altitude" in f.columns:
            alt = f["altitude"].values
        elif "Alt" in f.columns:
            alt = f["Alt"].values
        else:
            raise ValueError(
                "No altitude column found ('altitude' or 'Alt') while use_alt=True"
            )

    dy = np.abs(np.diff(y))
    stable = np.concatenate([[False], dy < tol])

    segments = []
    start = None
    for i, s in enumerate(stable):
        cond_alt = (not use_alt) or (alt[i] > alt_threshold)
        cond_abs = (min_abs_value is None) or (np.abs(y[i]) > min_abs_value)
        cond = s and cond_alt and cond_abs

        if cond and start is None:
            start = i
        elif (not cond) and start is not None:
            if i - start >= min_len:
                seg = {
                    "start_idx": start,
                    "end_idx": i - 1,
                    "start_time": time[start],
                    "end_time": time[i - 1],
                    "var_mean": np.mean(y[start:i]),
                }
                if use_alt:
                    seg["alt_mean"] = np.mean(alt[start:i])
                segments.append(seg)
            start = None

    if start is not None and len(time) - start >= min_len:
        seg = {
            "start_idx": start,
            "end_idx": len(time) - 1,
            "start_time": time[start],
            "end_time": time[-1],
            "var_mean": np.mean(y[start:]),
        }
        if use_alt:
            seg["alt_mean"] = np.mean(alt[start:])
        segments.append(seg)

    return segments, y


def add_segment_column(f, segments, col_name, fill_value=0.0):
    """Add a column populated with segment mean values elsewhere filled with a default.

    Args:
        f: Input DataFrame.
        segments: List of segment dictionaries with time boundaries and means.
        col_name: Name of the column to create.
        fill_value: Default value for non-segment rows.

    Returns:
        DataFrame with the added column.
    """
    f[col_name] = fill_value
    if len(segments) == 0:
        return f

    t = f["time"] if "time" in f.columns else f.index
    for seg in segments:
        mask = (t >= seg["start_time"]) & (t <= seg["end_time"])
        f.loc[mask, col_name] = seg["var_mean"]
    return f


def build_spd_and_vert_selected_from_segments(
    f: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
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

    if config.get("add_alt", False):
        alt_cfg = config.get("alt", {})
        alt_segments, _ = detect_constant_segments(f, var_name="altitude", **alt_cfg)
        f = add_segment_column(f, alt_segments, "selected_mcp", fill_value=np.nan)
        f.loc[f.index[-1], "selected_mcp"] = f.loc[f.index[-1], "altitude"]
        f["selected_mcp"] = f["selected_mcp"].bfill()

    return f


def isa_pressure(h_m: np.ndarray) -> np.ndarray:
    """Compute ISA static pressure (Pa) for altitude in meters."""
    T_tropo = T0 - 0.0065 * h_m
    p_tropo = p0 * (T_tropo / T0) ** (g / (R * 0.0065))
    T_strato = 216.65
    p_strato = 22632.06 * np.exp(-g * (h_m - 11000) / (R * T_strato))
    return np.where(h_m <= 11000, p_tropo, p_strato)


def compute_mach_and_cas(
    tas_kt: np.ndarray,
    alt_ft: np.ndarray,
    temp_K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mach and CAS from TAS [kt], altitude [ft], temperature [K].

    Args:
        tas_kt: True airspeed in knots.
        alt_ft: Altitude in feet.
        temp_K: Air temperature in Kelvin.

    Returns:
        Tuple of Mach number and calibrated airspeed in knots.
    """
    tas = np.asarray(tas_kt) * 0.514444  # kt → m/s
    h = np.asarray(alt_ft) * 0.3048  # ft → m

    a = np.sqrt(gamma_ratio * R * temp_K)
    mach = tas / a

    p = isa_pressure(h)

    pt_over_p = (1 + (gamma_ratio - 1) / 2 * mach**2) ** (
        gamma_ratio / (gamma_ratio - 1)
    )

    qc_p0 = (p / p0) * (pt_over_p - 1)

    cas = a0 * np.sqrt(
        (2 / (gamma_ratio - 1))
        * (((qc_p0 + 1) ** ((gamma_ratio - 1) / gamma_ratio)) - 1)
    )
    cas_kt = cas / 0.514444  # m/s → kt

    return mach, cas_kt


def compute_tas(df: pd.DataFrame) -> pd.Series:
    """Compute True Airspeed (TAS) from GS + wind components (output in knots).

    Args:
        df: DataFrame containing wind components and groundspeed.

    Returns:
        Series of TAS values in knots.
    """
    MS_TO_KT = 1.94384
    u_wind_kt = df["u_component_of_wind"] * MS_TO_KT
    v_wind_kt = df["v_component_of_wind"] * MS_TO_KT

    track_rad = np.deg2rad(df["track"])
    u_ground = df["groundspeed"] * np.sin(track_rad)
    v_ground = df["groundspeed"] * np.cos(track_rad)
    return np.sqrt((u_ground - u_wind_kt) ** 2 + (v_ground - v_wind_kt) ** 2)


def crop_on_distance_jump(
    f: pd.DataFrame,
    threshold: float = 200,
    min_speed: float = 90,
    upper_threshold: float = 3000,
) -> Tuple[pd.DataFrame, Any, Any]:
    """Crop flight data to remove segments with large distance jumps.

    Args:
        f: Flight DataFrame.
        threshold: Minimum jump (meters) considered a discontinuity.
        min_speed: Minimum groundspeed to retain data.
        upper_threshold: Upper bound to ignore extremely large jumps.

    Returns:
        Tuple of cropped DataFrame, ICAO identifier, and jump metric.
    """
    f2 = f.query("groundspeed > @min_speed").copy()

    mask = f2["distance_along_track_m"].diff() > threshold

    if mask.any():
        idx_first = mask.idxmax()
        idx_last = mask[::-1].idxmax()
        f3 = f2.loc[idx_first:idx_last].reset_index(drop=True)
    else:
        f3 = f2.reset_index(drop=True)
    icao_24 = f3.icao24.iloc[0]
    metric = len(f3) - len(
        f3[
            (f3["distance_along_track_m"].diff() > threshold)
            & (f3["distance_along_track_m"].diff() < upper_threshold)
        ]
    )
    return f3, icao_24, metric


def process_flight(
    f_id: Any,
    f: pd.DataFrame,
    output_dir_path: Any,
    selected_param_config: Dict[str, Any],
):
    """Process a single flight dataframe and persist if valid.

    Args:
        f_id: Flight identifier.
        f: Flight DataFrame.
        output_dir_path: Base directory where processed flights are stored.
        selected_param_config: Configuration for segment-based parameter selection.

    Returns:
        Tuple of flight id and success flag.
    """
    try:
        res = build_spd_and_vert_selected_from_segments(f, selected_param_config)
        res = add_cumulative_distance(res)
        res["long_wind"] = res["TAS"] - res["groundspeed"]
        res["gamma_air"] = np.arcsin((res["vertical_rate"] * ftmn) / (res["TAS"] * kt))

        typecode = res["typecode"].iloc[0]

        output_dir = output_dir_path / str(typecode)
        output_dir.mkdir(parents=True, exist_ok=True)

        res, icao_24, metric = crop_on_distance_jump(res)
        path = output_dir / f"{f_id}_{icao_24}_{metric}.parquet"

        if (res.adep_dist.isna().sum() == 0) and (res.ades_dist.isna().sum() == 0):
            res.to_parquet(path, index=False)
        return f_id, True

    except Exception as e:
        print(f"❌ Error for flight {f_id}: {e}")
        return f_id, False


def save_all_flights(
    df: pd.DataFrame,
    output_dir_path: Any,
    selected_param_config: Dict[str, Any],
    n_jobs: int = 8,
):
    """Process all flights in a DataFrame in parallel.

    Args:
        df: Input DataFrame containing multiple flights.
        output_dir_path: Output directory for processed parquet files.
        selected_param_config: Configuration for selected parameter detection.
        n_jobs: Number of parallel workers.

    Returns:
        List of tuples with flight id and success flag.
    """
    tasks = ((f_id, f) for f_id, f in df.groupby("flight_id"))
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(process_flight)(f_id, f, output_dir_path, selected_param_config)
        for f_id, f in tasks
    )
    return results


def haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Compute great-circle distance between coordinate pairs (meters).

    Args:
        lat1: Latitudes of first points in degrees.
        lon1: Longitudes of first points in degrees.
        lat2: Latitudes of second points in degrees.
        lon2: Longitudes of second points in degrees.

    Returns:
        Array of distances in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_cumulative_distance(
    df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"
) -> pd.DataFrame:
    """Add cumulative along-track distance column to a DataFrame.

    Args:
        df: Input DataFrame with latitude/longitude columns.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        DataFrame with `distance_along_track_m` appended.
    """

    d = haversine(
        df[lat_col].iloc[:-1].values,
        df[lon_col].iloc[:-1].values,
        df[lat_col].iloc[1:].values,
        df[lon_col].iloc[1:].values,
    )

    cumulative_d = np.concatenate(([0], np.cumsum(d)))

    df = df.copy()
    df["distance_along_track_m"] = cumulative_d
    return df


def process_files(
    arco_grid: Any,
    file_path: Any,
    output_dir_path: Any,
    selected_param_config: Dict[str, Any],
):
    """Process a parquet file through interpolation, TAS/CAS computation, and per-flight export.

    Args:
        arco_grid: Interpolator object providing `interpolate` method.
        file_path: Path to the parquet flight file.
        output_dir_path: Directory where processed flights are stored.
        selected_param_config: Configuration for segment-based parameter selection.

    Returns:
        None. Writes processed parquet files and logs progress.
    """
    df = pd.read_parquet(file_path)

    df = df.drop(
        columns=[
            "bds05",
            "bds18",
            "bds19",
            "bds21",
            "selected_fms",
            "target_source",
        ],
        errors="ignore",
    ).drop_duplicates()

    df = arco_grid.interpolate(df)
    df["TAS"] = compute_tas(df)
    df["Mach"], df["CAS"] = compute_mach_and_cas(
        df["TAS"], df["altitude"], df["temperature"]
    )
    results = save_all_flights(df, output_dir_path, selected_param_config, n_jobs=20)
    print("✅ Done.")
    print(pd.DataFrame(results, columns=["flight_id", "success"]))
