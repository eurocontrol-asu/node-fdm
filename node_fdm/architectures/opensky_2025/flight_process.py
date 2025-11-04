from node_fdm.architectures.opensky_2025.columns import col_alt_diff, col_alt_sel, col_alt, col_dist

LOW_THR = 200  # meters
UPPER_THR = 3000  # meters

def flight_processing(df):
    """
    User-defined custom step for flight preprocessing.
    """
    df[col_alt_diff] = df[col_alt_sel] - df[col_alt]

    return df


def segment_filtering(f, start_idx, seq_len):
    """
    User-defined custom step for segment filtering.
    """
    dist_diff = f[col_dist].diff(1)
    seg = dist_diff.iloc[start_idx : start_idx + seq_len]
    condition = len(seg[(seg < LOW_THR) | (seg > UPPER_THR)]) == 0
    return condition


selected_param_config = {
    "mach": {"tol": 0.002, "min_len": 120, "alt_threshold": 20000, "use_alt": True},
    "cas":  {"tol": 1.0, "min_len": 60, "use_alt": False, "smooth_window": 15, "smooth_method": "savgol"},
    "vz":   {"tol": 25, "min_len": 30, "use_alt": False, "min_abs_value": 50, "smooth_window": 15, "smooth_method": "savgol"},
    "add_alt": False
}