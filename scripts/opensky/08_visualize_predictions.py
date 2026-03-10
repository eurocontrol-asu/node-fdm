# %%
"""08 — Visualize predictions vs truth vs BADA for sample flights."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import yaml

from node_fdm_bada.utils import cas_to_mach, tas_to_cas
from node_fdm_data.preprocessing.opensky import flight_processing
from node_fdm_data.processor import FlightProcessor


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    process_dir = data_dir / cfg["paths"]["process_dir"]
    predict_dir = data_dir / cfg["paths"]["predicted_dir"]
    bada_dir = data_dir / cfg["paths"]["bada_dir"]

    processor = FlightProcessor(steps=[flight_processing])
    split_df = pl.read_csv(process_dir / "dataset_split.csv")
    acft = "A320"

    data_df = split_df.filter(pl.col("aircraft_type") == acft)
    test_df = data_df.filter(pl.col("split") == "test")
    node_pred_dir = predict_dir / acft
    output_dir = bada_dir / acft

    for row in test_df.iter_rows(named=True):
        flight_path = row["filepath"]
        fname = Path(flight_path).name
        if not (output_dir / fname).exists():
            continue

        f = pl.read_parquet(flight_path)
        processor.process(f).collect()  # validate processing
        f2 = pl.read_parquet(node_pred_dir / fname)
        f = f.hstack(f2)
        f3 = pl.read_parquet(output_dir / fname)
        f = f.hstack(f3)

        # Compute derived CAS/Mach for BADA
        bada_cas_ms = tas_to_cas(
            f["bada_tas_ms"].to_numpy(),
            f["bada_alt_std_m"].to_numpy(),
            f["temp_k"].to_numpy(),
        )
        bada_mach = cas_to_mach(bada_cas_ms, f["bada_alt_std_m"].to_numpy())
        f = f.with_columns(
            pl.Series("bada_cas_ms", bada_cas_ms),
            pl.Series("bada_mach", bada_mach),
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.flatten()

        # Altitude
        axes[0].plot(f["alt_std_m"], color="r", label="True")
        axes[0].plot(f["pred_alt_std_m"], color="b", label="Pred")
        axes[0].plot(f["bada_alt_std_m"], color="g", label="Bada")
        axes[0].plot(f["alt_sel_m"], "--", color="k", lw=0.5, label="Selected")
        axes[0].set_title("Altitude [m]")

        # TAS
        axes[1].plot(f["tas_ms"], color="r", label="True")
        axes[1].plot(f["pred_tas_ms"], color="b", label="Pred")
        axes[1].plot(f["bada_tas_ms"], color="g", label="Bada")
        axes[1].set_title("True Airspeed [m/s]")

        # Gamma
        axes[2].plot(f["gamma_rad"], color="r", label="True")
        axes[2].plot(f["pred_gamma_rad"], color="b", label="Pred")
        axes[2].plot(f["bada_gamma_rad"], color="g", label="Bada")
        axes[2].set_title("Flight path angle gamma [rad]")

        for ax in axes:
            ax.grid(True, lw=0.3, linestyle="--", alpha=0.7)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

        f.write_parquet(data_dir / "example.parquet")
        break


if __name__ == "__main__":
    main()
# %%
