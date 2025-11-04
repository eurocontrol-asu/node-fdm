import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


from joblib import Parallel, delayed

from utils.physics.constants import (
    T0,
    p0,
    g,
    R,
    gamma_ratio,
    a0, 
    ftmn,
    kt
)


def detect_constant_segments(
    f, var_name, tol=0.002, min_len=5, alt_threshold=20000, use_alt=True,
    min_abs_value=None, smooth_window=None, smooth_method="rolling"
):
    """
    Détecte les segments à valeur quasi constante pour une variable donnée.
    Retourne (segments, y_smooth).
    """
    if var_name not in f.columns:
        raise ValueError(f"Variable '{var_name}' not found in DataFrame")

    if "time" in f.columns:
        time = f["time"].values
    else:
        time = f.index.values

    y = f[var_name].astype(float).values

    # Lissage optionnel
    if smooth_window is not None and smooth_window > 1:
        if smooth_method == "rolling":
            y = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean().values
        elif smooth_method == "savgol":
            win = min(smooth_window, len(y) - (len(y) % 2 == 0))
            y = savgol_filter(y, window_length=win, polyorder=2, mode="interp")

    # Altitude
    alt = None
    if use_alt:
        if "altitude" in f.columns:
            alt = f["altitude"].values
        elif "Alt" in f.columns:
            alt = f["Alt"].values
        else:
            raise ValueError("No altitude column found ('altitude' or 'Alt') while use_alt=True")

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
    """Ajoute une colonne avec la valeur moyenne du segment, sinon 0."""
    f[col_name] = fill_value
    if len(segments) == 0:
        return f

    t = f["time"] if "time" in f.columns else f.index
    for seg in segments:
        mask = (t >= seg["start_time"]) & (t <= seg["end_time"])
        f.loc[mask, col_name] = seg["var_mean"]
    return f


def build_spd_and_vert_selected_from_segments(f, config):
    """
    Build selected variables (Mach, CAS, vertical_rate, altitude) from constant-segment detection.
    Configuration is provided through the `config` dictionary, e.g.:

    config = {
        "mach": {"tol": 0.002, "min_len": 120, "alt_threshold": 20000, "use_alt": True},
        "cas":  {"tol": 1.0, "min_len": 60, "use_alt": False, "smooth_window": 15, "smooth_method": "savgol"},
        "vz":   {"tol": 25, "min_len": 30, "use_alt": False, "min_abs_value": 50, "smooth_window": 15, "smooth_method": "savgol"},
        "add_alt": False
    }
    """
    f = f.copy()

    # 1️⃣ Mach constant segments
    mach_cfg = config.get("mach", {})
    mach_segments, _ = detect_constant_segments(f, var_name="Mach", **mach_cfg)
    f = add_segment_column(f, mach_segments, "mach_sel")

    # 2️⃣ CAS segments (excluding Mach periods)
    f_cas = f.copy()
    t = f["time"] if "time" in f.columns else f.index
    if len(mach_segments) > 0:
        for seg in mach_segments:
            mask = (t >= seg["start_time"]) & (t <= seg["end_time"])
            f_cas.loc[mask, "CAS"] = np.nan

    cas_cfg = config.get("cas", {})
    cas_segments, _ = detect_constant_segments(f_cas, var_name="CAS", **cas_cfg)
    f = add_segment_column(f, cas_segments, "cas_sel")

    # 3️⃣ Vertical rate segments
    vz_cfg = config.get("vz", {})
    vz_segments, _ = detect_constant_segments(f, var_name="vertical_rate", **vz_cfg)
    f = add_segment_column(f, vz_segments, "vz_sel")

    # 4️⃣ Altitude (optional)
    if config.get("add_alt", False):
        alt_cfg = config.get("alt", {})
        alt_segments, _ = detect_constant_segments(f, var_name="altitude", **alt_cfg)
        f = add_segment_column(f, alt_segments, "selected_mcp", fill_value=np.nan)
        f.loc[f.index[-1], "selected_mcp"] = f.loc[f.index[-1], "altitude"]
        f["selected_mcp"] = f["selected_mcp"].bfill()

    return f



def isa_pressure(h_m):
    """Pression statique selon ISA (Pa) pour altitude en mètres."""
    T_tropo = T0 - 0.0065 * h_m
    p_tropo = p0 * (T_tropo / T0) ** (g / (R * 0.0065))
    T_strato = 216.65
    p_strato = 22632.06 * np.exp(-g * (h_m - 11000) / (R * T_strato))
    return np.where(h_m <= 11000, p_tropo, p_strato)

def compute_mach_and_cas(tas_kt, alt_ft, temp_K):
    """
    Calcule Mach et CAS à partir de TAS [kt], altitude [ft], température [K].
    Résultats vectorisés (np.array ou pd.Series).
    """
    # Conversion en unités SI
    tas = np.asarray(tas_kt) * 0.514444   # kt → m/s
    h = np.asarray(alt_ft) * 0.3048       # ft → m
    
    # Vitesse du son à température réelle
    a = np.sqrt(gamma_ratio * R * temp_K)
    mach = tas / a

    # Pression statique ISA
    p = isa_pressure(h)
    
    # Pression totale
    pt_over_p = (1 + (gamma_ratio - 1) / 2 * mach**2) ** (gamma_ratio / (gamma_ratio - 1))
    
    # Pression dynamique ramenée au niveau de la mer
    qc_p0 = (p / p0) * (pt_over_p - 1)
    
    # CAS (Calibrated Airspeed)
    cas = a0 * np.sqrt((2 / (gamma_ratio - 1)) * (((qc_p0 + 1) ** ((gamma_ratio - 1) / gamma_ratio)) - 1))
    cas_kt = cas / 0.514444  # m/s → kt
    
    return mach, cas_kt


def compute_tas(df: pd.DataFrame) -> pd.Series:
    """Compute True Airspeed (TAS) from GS + wind components (output in knots)."""
    MS_TO_KT = 1.94384
    u_wind_kt = df["u_component_of_wind"] * MS_TO_KT
    v_wind_kt = df["v_component_of_wind"] * MS_TO_KT

    track_rad = np.deg2rad(df["track"])
    u_ground = df["groundspeed"] * np.sin(track_rad)
    v_ground = df["groundspeed"] * np.cos(track_rad)
    return np.sqrt((u_ground - u_wind_kt)**2 + (v_ground - v_wind_kt)**2)


def crop_on_distance_jump(f, threshold=200, min_speed=90, upper_threshold=3000):
    # Filter by groundspeed
    f2 = f.query("groundspeed > @min_speed").copy()

    # Detect large distance jumps
    mask = f2["distance_along_track_m"].diff() > threshold

    # Crop between first and last jump
    if mask.any():
        idx_first = mask.idxmax()
        idx_last = mask[::-1].idxmax()
        f3 = f2.loc[idx_first:idx_last].reset_index(drop=True)
    else:
        f3 = f2.reset_index(drop=True)
    icao_24 = f3.icao24.iloc[0]
    metric = len(f3)- len(f3[(f3["distance_along_track_m"].diff() > threshold) & (f3["distance_along_track_m"].diff() < upper_threshold)])
    return f3, icao_24, metric


def process_flight(f_id, f, output_dir_path, selected_param_config):
    try:
        res = build_spd_and_vert_selected_from_segments(f, selected_param_config)
        res = add_cumulative_distance(res)
        res["long_wind"] = res["TAS"] - res["groundspeed"]
        res['gamma_air'] = np.arcsin( (res["vertical_rate"] * ftmn) / (res["TAS"] * kt))
        
        typecode = res["typecode"].iloc[0]

        # Build output folder and file path
        output_dir = output_dir_path / str(typecode)
        output_dir.mkdir(parents=True, exist_ok=True)



        res, icao_24, metric = crop_on_distance_jump(res)
        path = output_dir / f"{f_id}_{icao_24}_{metric}.parquet"

        

        # Save the file
        if (res.adep_dist.isna().sum() == 0) and (res.ades_dist.isna().sum() == 0):
            res.to_parquet(path, index=False)
        return f_id, True

    except Exception as e:
        print(f"❌ Error for flight {f_id}: {e}")
        return f_id, False
    

def save_all_flights(df, output_dir_path, selected_param_config, n_jobs=8):
    tasks = ((f_id, f) for f_id, f in df.groupby("flight_id"))
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(process_flight)(f_id, f, output_dir_path, selected_param_config) for f_id, f in tasks
    )
    return results



def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def add_cumulative_distance(df, lat_col='latitude', lon_col='longitude'):
    # Compute segment distances between consecutive points
    d = haversine(df[lat_col].iloc[:-1].values,
                  df[lon_col].iloc[:-1].values,
                  df[lat_col].iloc[1:].values,
                  df[lon_col].iloc[1:].values)
    # Build cumulative distance array (start at 0)
    cumulative_d = np.concatenate(([0], np.cumsum(d)))
    # Add to DataFrame
    df = df.copy()
    df['distance_along_track_m'] = cumulative_d
    return df


def process_files(arco_grid, file_path, output_dir_path, selected_param_config):
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
        errors="ignore"   
    ).drop_duplicates()
        
    df = arco_grid.interpolate(df)
    df["TAS"] = compute_tas(df)
    df["Mach"], df["CAS"] = compute_mach_and_cas(df["TAS"], df["altitude"], df["temperature"])
    results = save_all_flights(df, output_dir_path, selected_param_config, n_jobs=20)
    print("✅ Done.")
    print(pd.DataFrame(results, columns=["flight_id", "success"]))