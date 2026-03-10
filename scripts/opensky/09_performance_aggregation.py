# %%
"""09 — Aggregate prediction performance metrics by flight phase."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml
from tqdm import tqdm

from node_fdm_data.preprocessing.opensky import flight_processing
from node_fdm_data.processor import FlightProcessor


def compute_errors_by_phase(
    df: pl.DataFrame,
    pred_col: str,
    target_col: str,
    vertical_rate_col: str = "vz_ms",
    eps: float = 1e-8,
) -> pl.DataFrame:
    """Compute MAE, MAPE, ME and their std by flight phase."""
    vz = df[vertical_rate_col].to_numpy()
    climb_mask = vz > 1.0
    descent_mask = vz < -1.0
    level_mask = (~climb_mask) & (~descent_mask)

    phases = {
        "All phases": np.ones(len(df), dtype=bool),
        "Climb": climb_mask,
        "Level flight": level_mask,
        "Descent": descent_mask,
    }

    is_angle = "gamma" in pred_col.lower()
    results = []

    for phase, mask in phases.items():
        y_pred = df.filter(pl.Series(mask))[pred_col].to_numpy()
        y_true = df.filter(pl.Series(mask))[target_col].to_numpy()

        valid = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[valid], y_true[valid]
        if len(y_true) == 0:
            continue

        if is_angle:
            deg_factor = 180 / np.pi
            y_pred = y_pred * deg_factor
            y_true = y_true * deg_factor

        err = y_pred - y_true
        abs_err = np.abs(err)

        if is_angle:
            abs_perc_err = np.full_like(abs_err, np.nan)
        else:
            abs_perc_err = np.abs(err / (y_true + eps)) * 100.0

        results.append(
            (
                phase,
                float(np.mean(abs_err)),
                float(np.std(abs_err)),
                float(np.nanmean(abs_perc_err)),
                float(np.nanstd(abs_perc_err)),
                float(np.mean(err)),
                float(np.std(err)),
                len(y_true),
            )
        )

    return pl.DataFrame(
        results,
        schema=[
            ("Phase", pl.Utf8),
            ("MAE", pl.Float64),
            ("MAE_std", pl.Float64),
            ("MAPE (%)", pl.Float64),
            ("MAPE_std", pl.Float64),
            ("ME", pl.Float64),
            ("ME_std", pl.Float64),
            ("Count", pl.Int64),
        ],
        orient="row",
    )


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    process_dir = data_dir / cfg["paths"]["process_dir"]
    predict_dir = data_dir / cfg["paths"]["predicted_dir"]
    bada_dir = data_dir / cfg["paths"]["bada_dir"]

    typecodes = cfg["typecodes"]
    processor = FlightProcessor(steps=[flight_processing])

    variables = {
        "alt_std_m": "Altitude [m]",
        "tas_ms": "True airspeed [m/s]",
        "gamma_rad": "Flight path angle [deg]",
    }

    all_results: list[pl.DataFrame] = []

    for acft in typecodes:
        acft_dir = bada_dir / acft
        if not acft_dir.exists():
            continue
        parquet_files = sorted(acft_dir.glob("*.parquet"))
        print(f"Processing {acft}: {len(parquet_files)} flights")

        acft_frames: list[pl.DataFrame] = []
        for file in tqdm(parquet_files, desc=f"{acft} flights"):
            f = pl.read_parquet(process_dir / acft / file.name)
            processor.process(f).collect()  # validate processing
            f2 = pl.read_parquet(predict_dir / acft / file.name)
            f3 = pl.read_parquet(bada_dir / acft / file.name)
            f = f.hstack(f2).hstack(f3)
            f = f.filter(pl.col("altitude") > 5000)

            n_unique_alt_sel = f["alt_sel_m"].n_unique()
            if n_unique_alt_sel > 3:
                last_alt_sel = f["alt_sel_m"][-1]
                if last_alt_sel > 5000:
                    diff_abs = (pl.col("alt_sel_m") - pl.col("alt_std_m")).abs()
                    mask = f.select(diff_abs > 5000).to_series()
                    # Find last False from end
                    inv_mask = (~mask).to_list()[::-1]
                    pos = next((i for i, v in enumerate(inv_mask) if v), len(inv_mask))
                    pos_from_start = len(mask) - 1 - pos
                    f = f.head(pos_from_start)

                lat_diff = f["latitude"].diff(1).abs().max()
                dist_diff = f["distance_along_track_m"].diff(1).abs().max()
                if lat_diff < 0.3 and dist_diff < 10000:
                    acft_frames.append(f)

        if not acft_frames:
            continue

        df_acft = pl.concat(acft_frames, how="vertical")

        for var, label in variables.items():
            for prefix in ["bada_", "pred_"]:
                pred_col = f"{prefix}{var}"
                if pred_col not in df_acft.columns or var not in df_acft.columns:
                    print(f"⚠️ Missing {var} or {pred_col} in {acft}, skipping.")
                    continue

                metrics = compute_errors_by_phase(df_acft, pred_col=pred_col, target_col=var)
                metrics = metrics.with_columns(
                    pl.lit(acft).alias("Aircraft"),
                    pl.lit(label).alias("Variable"),
                    pl.lit(prefix[:-1].upper()).alias("Model"),
                )
                all_results.append(metrics)

    # === Combine ===
    final_df = pl.concat(all_results, how="vertical")
    final_df = final_df.select(
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
    )

    phase_order = {"All phases": 0, "Climb": 1, "Level flight": 2, "Descent": 3}
    final_df = (
        final_df.with_columns(
            pl.col("Phase").replace_strict(phase_order, default=99).alias("_phase_order")
        )
        .sort("Aircraft", "Variable", "_phase_order", "Model")
        .drop("_phase_order")
    )

    final_df.write_parquet(data_dir / "performance.parquet")
    print(f"✅ Saved performance.parquet ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
# %%
