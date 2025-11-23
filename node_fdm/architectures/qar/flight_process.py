import numpy as np
import pandas as pd 

from scipy.signal import butter, filtfilt

from utils.physics.constants import (
    gamma_ratio,
    R,
)

from node_fdm.architectures.qar.model import X_COLS

from node_fdm.architectures.qar.columns import (
    col_alt_diff, 
    col_spd_diff,
    col_alt_sel, 
    col_spd_sel,
    col_cas,
    col_alt,
    col_gs,
    col_mach,
    col_temp,
    col_tas,
    col_gamma,
    col_on_ground,
    col_n1_left,
    col_n1_right,
    col_ff_left,
    col_ff_right,
    col_reduce_engine,
    col_ff,
    col_n1,
    col_vz_sel,
    col_runway_elev,
    col_fma_2,
    col_dist_to_thr,
    col_mass,
)

def mode_stabilize(series, min_duration=10):
    stable_series = series.copy()
    current_mode = stable_series.iloc[0]
    count = 0
    for i in range(1, len(stable_series)):
        if stable_series.iloc[i] == current_mode:
            count = 0
        else:
            count += 1
            if count < min_duration:
                stable_series.iloc[i] = current_mode
            else:
                current_mode = stable_series.iloc[i]
                count = 0
    return stable_series


def one_reduce_eng_value(col_left, col_right):
    def one_reduce_value_curry(el):
        if el[col_reduce_engine]:
            return max(el[col_left], el[col_right])
        else:
            return (el[col_left]+ el[col_right]) / 2
    return one_reduce_value_curry

def reduce_engine_vectorized(df, col_left, col_right, col_reduce_engine):
    mask = df[col_reduce_engine] != 0
    result = pd.Series(index=df.index, dtype=float)
    result[mask] = df[[col_left, col_right]].loc[mask].max(axis=1)
    result[~mask] = df[[col_left, col_right]].loc[~mask].mean(axis=1)
    
    return result

def engine_process(df):
    reduce = (df[col_n1_left] < 5) | (df[col_n1_right] < 5)
    diff =  np.abs(df[col_n1_left] - df[col_n1_right]) > 5
    df[col_reduce_engine] = (reduce * diff).astype(int)
    df[col_ff] = df[col_ff_left] + df[col_ff_right]
    df[col_n1] = reduce_engine_vectorized(df, col_n1_left, col_n1_right, col_reduce_engine)
    return df

def smooth_strong(x, window_size=100):
    x = np.asarray(x, dtype=float)
    half = window_size // 2
    
    x_padded = np.pad(x, pad_width=(half, half-1), mode='reflect')
    
    kernel = np.ones(window_size) / window_size
    y = np.convolve(x_padded, kernel, mode='valid')
    
    return y

def filter_noise(df_col, fs=1.0, cutoff=0.04, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    signal_filtre = filtfilt(b, a, df_col)
    return signal_filtre

def flight_processing(df, step=4):
    """
    User-defined custom step for flight preprocessing.
    """
    df[col_tas] = df[col_tas].fillna(df[col_gs])
    speed_of_sound = np.sqrt(gamma_ratio * R * df[col_temp])
    df[col_mach] = df[col_tas] / speed_of_sound
    
    df[col_gamma] = np.arcsin(df[col_gs] * np.sin(df[col_gamma]) / df[col_tas]) # Compute Gamma_air with no vertical wind assumption
    df[col_gamma] = df[col_gamma].fillna(0)
            
    smooth_gamma_rad = smooth_strong(df[col_gamma])
    smooth_vz_sel = np.tan(smooth_gamma_rad) * df[col_gs]
    df[col_vz_sel] =  mode_stabilize(df[col_vz_sel])
    df[col_vz_sel] = df.fma_col_2_category.isin([6, 11, 12, 13, 14, 15]) * smooth_vz_sel + df.fma_col_2_category.isin([10]) * df[col_vz_sel]

    alt_cond1 = df[col_alt].diff(5).fillna(0)>500
    alt_cond2 = df[col_alt].diff(-5).fillna(0)>500
    prod_cond = alt_cond1 * alt_cond2
    prod_cond = prod_cond.apply(lambda el : np.nan if el==1 else 1)
    df[col_alt] = df[col_alt] * prod_cond
    df[col_alt] = df[col_alt].where(df[col_on_ground]==0).bfill().ffill()
    
    
    df[col_runway_elev] = df[col_alt].where(df[col_on_ground]==1).bfill()
    df[col_alt_sel] =  np.where(df[col_fma_2].isin([12, 13, 14, 15]), df[col_runway_elev], df[col_alt_sel])  
    df[col_alt_sel] =  np.where(df[col_dist_to_thr]<3, df[col_runway_elev], df[col_alt_sel])  
    
    df = engine_process(df)

    df[col_mass.derivative] = -df[col_ff]
    
    df[col_alt_diff] = df[col_alt_sel] - df[col_alt]
    df[col_spd_diff] = df[col_spd_sel] - df[col_cas]
    
    
    df = df.bfill().ffill()
    df = df[df[col_on_ground] == 0]
    df = df.iloc[::step]
    return df


def segment_filtering(f, start_idx, seq_len):
    """
    User-defined custom step for segment filtering.
    """
    return True
