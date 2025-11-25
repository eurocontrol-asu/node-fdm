# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1]
sys.path.append(str(root_path))

import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config import DATA_DIR, PROCESS_DIR, PREDICT_DIR, BADA_DIR, TYPECODES

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS
from node_fdm.architectures.opensky_2025.flight_process import flight_processing


processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)

# For clean decimal display
pd.options.display.float_format = "{:.2f}".format


def compute_errors_by_phase(
    df, pred_col, target_col, vertical_rate_col="vz_ms", eps=1e-8
):
    """Compute MAE, MAPE, ME and their std by phase, with optional rad→deg conversion."""
    climb_mask = df[vertical_rate_col] > 1.0
    descent_mask = df[vertical_rate_col] < -1.0
    level_mask = (~climb_mask) & (~descent_mask)

    phases = {
        "All phases": np.ones(len(df), dtype=bool),
        "Climb": climb_mask,
        "Level flight": level_mask,
        "Descent": descent_mask,
    }

    # detect angular variable
    is_angle = "gamma" in pred_col.lower()

    results = []
    for phase, mask in phases.items():
        sub = df[mask]
        if sub.empty:
            continue

        y_pred = sub[pred_col].to_numpy()
        y_true = sub[target_col].to_numpy()

        valid = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[valid], y_true[valid]
        if len(y_true) == 0:
            continue

        # Convert radians to degrees if angular variable
        if is_angle:
            deg_factor = 180 / np.pi
            y_pred = y_pred * deg_factor
            y_true = y_true * deg_factor

        err = y_pred - y_true
        abs_err = np.abs(err)

        # For gamma, MAPE has no physical meaning → fill with NaN
        if is_angle:
            abs_perc_err = np.full_like(abs_err, np.nan)
        else:
            abs_perc_err = np.abs(err / (y_true + eps)) * 100.0

        mae_mean, mae_std = np.mean(abs_err), np.std(abs_err)
        mape_mean, mape_std = np.nanmean(abs_perc_err), np.nanstd(abs_perc_err)
        me_mean, me_std = np.mean(err), np.std(err)
        count = len(y_true)

        results.append(
            (phase, mae_mean, mae_std, mape_mean, mape_std, me_mean, me_std, count)
        )

    return pd.DataFrame(
        results,
        columns=[
            "Phase",
            "MAE",
            "MAE_std",
            "MAPE (%)",
            "MAPE_std",
            "ME",
            "ME_std",
            "Count",
        ],
    ).set_index("Phase")


variables = {
    "alt_std_m": "Altitude [m]",
    "tas_ms": "True airspeed [m/s]",
    "gamma_rad": "Flight path angle [deg]",  # handled specially
}
all_results = []

for acft in TYPECODES:
    acft_records = []
    acft_dir = BADA_DIR / acft
    if not os.path.exists(acft_dir):
        continue
    files = os.listdir(BADA_DIR / acft)
    print(f"Processing {acft}: {len(files)} flights")

    for file in tqdm(files, desc=f"{acft} flights"):

        f = pd.read_parquet(PROCESS_DIR / acft / file)
        f1 = processor.process_flight(f)
        f2 = pd.read_parquet(PREDICT_DIR / acft / file)
        f3 = pd.read_parquet(BADA_DIR / acft / file)
        f = f.join(f2).join(f3)
        f = f[f.altitude > 5000]
        if len(f.alt_sel_m.unique()) > 3:
            if f.alt_sel_m.iloc[-1] > 5000:
                mask = (f.alt_sel_m - f.alt_std_m).abs() > 5000
                pos = np.where(~mask.values[::-1])[0][0]
                pos_from_start = len(mask) - 1 - pos
                f = f.iloc[:pos_from_start]
            diff = f["latitude"].diff(1).abs().max()
            diff2 = f.distance_along_track_m.diff(1).abs().max()
            if (diff < 0.3) & (diff2 < 10000):
                acft_records.append(f)

    if not acft_records:
        continue

    df_acft = pd.concat(acft_records, ignore_index=True)

    for var, label in variables.items():
        # Compare both BADA and predicted model outputs against the true values
        for prefix in ["bada_", "pred_"]:
            pred_col = f"{prefix}{var}"
            target_col = var

            if pred_col not in df_acft.columns or target_col not in df_acft.columns:
                print(f"⚠️ Missing {var} or {pred_col} in {acft}, skipping.")
                continue

            metrics = compute_errors_by_phase(
                df_acft, pred_col=pred_col, target_col=target_col
            )
            metrics["Aircraft"] = acft
            metrics["Variable"] = label
            metrics["Model"] = prefix[:-1].upper()  # “BADA” or “PRED”
            all_results.append(metrics.reset_index())


# === Combine all results ===
final_df = pd.concat(all_results, ignore_index=True)
final_df = final_df[
    [
        "Aircraft",
        "Variable",
        "Phase",
        "Model",
        "MAE",
        "MAE_std",
        "MAPE (%)",
        "MAPE_std",
        "ME",
        "ME_std",
        "Count",
    ]
]

phase_order = ["All phases", "Climb", "Level flight", "Descent"]
final_df["Phase"] = pd.Categorical(
    final_df["Phase"], categories=phase_order, ordered=True
)
final_df = final_df.sort_values(
    by=["Aircraft", "Variable", "Phase", "Model"]
).reset_index(drop=True)
final_df = final_df.round(2)

final_df

from config import DATA_DIR

final_df.to_parquet(DATA_DIR / "performance.parquet", index=False)

# %%

final_df[final_df.Aircraft == "A319"]
# %%
