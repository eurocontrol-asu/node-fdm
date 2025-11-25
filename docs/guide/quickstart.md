# Quickstart

This guide provides a complete overview of how to run **node-fdm** end-to-end. It provide the general workflow used by all architectures and the full OpenSky 2025 pipeline example. All paths assume you are at the repository root (where `config.py` lives).

### General Pattern (Any Architecture)

1) **Collect and prepare raw data**  
   Ensure your raw inputs can be mapped to the architecture’s `Column` definitions.

2) **Decode, resample, and clean**  
   Build consistent time steps, decode dependent messages, and remove invalid or unusable segments.

3) **Feature enrichment**  
   Add environmental inputs (e.g., ERA5), smoothing, and architecture-specific derived quantities.

4) **Dataset splitting**  
   Create a file list with `train`, `val`, and `test` assignments pointing to processed parquet files.

5) **Training**  
   Use `ODETrainer`, setting `architecture_name` to your target architecture and loading its `model_params` from `model.py`.

6) **Inference**  
   Load checkpoints with `NodeFDMPredictor`, reuse the architecture’s preprocessing hooks, and write prediction parquet files.

7) **Evaluation and visualization**  
   Compute metrics and generate overlays for the output columns relevant to your architecture.

---

### OpenSky 2025 (ADS-B) Example Pipeline

1) **Aircraft sampling**  
   `code/opensky/01_aircraft_list.py` builds `data/aircraft_db.csv` using OpenSky Trino and `config.TYPECODES`.

2) **Download raw data**  
   `code/opensky/02_download_data.py` fetches history, flightlist, and extended tables into `data/downloaded_parquet/`.

3) **Decode and resample**  
   `code/opensky/03_preprocess_data.py <history.parquet>` decodes BDS 4/5/6, filters short flights, computes ADEP/ADES distances, and resamples to 4 seconds.

4) **Weather enrichment and smoothing**  
   `code/opensky/04_weather_spd_process_data.py` adds ERA5 wind/temperature (`ERA5_FEATURES`), applies `selected_param_config`, and writes enriched segments to  
   `data/processed_flights/<TYPECODE>/` plus `dataset_split.csv`.

5) **Train Neural ODEs**  
   `code/opensky/05_training.py` trains the `opensky_2025` architecture via `ODETrainer`, saving checkpoints to `models/opensky_<TYPECODE>/`.

6) **Inference**  
   `code/opensky/06_flight_prediction.py` rolls out trajectories and writes `pred_*` columns to `data/predicted_flights/<TYPECODE>/`.

7) **Baselines and evaluation**  
   - `07_bada_prediction.py` (requires `BADA_4_2_DIR`)  
   - `08_visualize_predictions.py` (overlays for truth vs. model vs. selections vs. BADA)  
   - `09_performance_aggregation.py` (MAE/MAPE/ME per phase)  
   - `10_dataset_stats.py` (coverage statistics)

---

### General Tips

- Use `config.py` as the single source of paths, typecodes, architecture names, and shared parameters.  
- Remove hard-coded overrides (e.g., `acft = "A320"` in `05_training.py`) if you want to train all types in a loop.  
- Remove the temporary `break` in `04_weather_spd_process_data.py` to process *all* preprocessed parquet files.  
- Make sure caches (e.g., `data/era5_cache`) exist to avoid repeated downloads of ERA5 fields.  
- For hardware constraints, adjust `batch_size`, `num_workers`, and `seq_len` inside `model_config`.

