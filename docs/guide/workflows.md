# End-to-end workflows

## OpenSky 2025 (ADS-B) pipeline
1) **Aircraft sampling** — `code/opensky/01_aircraft_list.py`  
   Builds `data/aircraft_db.csv` using OpenSky Trino + the typecodes in `config.TYPECODES`.

2) **Download raw data** — `code/opensky/02_download_data.py`  
   Pulls history, flightlist, and extended tables for sampled aircraft into `data/downloaded_parquet/`.

3) **Decode & resample** — `code/opensky/03_preprocess_data.py <history.parquet>`  
   - Decodes BDS 4/5/6, filters short flights, merges registration/typecode.  
   - Adds along-track distances to ADEP/ADES and resamples to 4s.

4) **Weather + smoothing** — `code/opensky/04_weather_spd_process_data.py`  
   - Uses `fastmeteo.ArcoEra5` to add wind/temperature (`ERA5_FEATURES`).  
   - Applies `selected_param_config` (smoothing and tolerance for selected Mach/CAS/VZ).  
   - Writes enriched flights to `data/processed_flights/<typecode>/` and `dataset_split.csv`.

5) **Train Neural ODEs** — `code/opensky/05_training.py`  
   - Architecture: `node_fdm.architectures.opensky_2025` (state: distance, altitude, gamma, TAS).  
   - Trainer: `ODETrainer` with `seq_len=60`, `step=4s`, AdamW, and modular checkpoints per layer.  
   - Output: `models/opensky_<TYPECODE>/meta.json` + `trajectory.pt` and `data_ode.pt`.

6) **Inference** — `code/opensky/06_flight_prediction.py`  
   Loads processed flights, rolls out trajectories with `NodeFDMPredictor`, and stores `pred_*` columns.

7) **Baselines and evaluation**  
   - BADA: `code/opensky/07_bada_prediction.py` (requires `BADA_4_2_DIR` + mapping).  
   - Visuals: `08_visualize_predictions.py` for quick overlays.  
   - Metrics: `09_performance_aggregation.py` computes MAE/MAPE/ME by phase and saves `performance.parquet`.  
   - Coverage: `10_dataset_stats.py` prints hours per split.

## General tips
- Keep `config.py` as the single source of paths and typecodes; adjust there instead of editing scripts inline.  
- Remove hard-coded overrides (e.g., `acft = "A320"` in `05_training.py`) when you want to train all types.  
- Remove the safety `break` in `04_weather_spd_process_data.py` if you want to process every preprocessed parquet (it currently stops after the first).  
- Ensure ERA5 cache exists (`data/era5_cache`) to avoid repeated downloads.  
- For GPU-constrained setups, lower `batch_size` and `num_workers` in `model_config`.
