# End-to-end workflows

## General pattern (any architecture)
1) **Collect/prepare raw data** — align fields so they can be converted into your `Column` definitions.  
2) **Decode / resample / clean** — build consistent time steps and remove invalid segments.  
3) **Feature enrich** — add environment variables (e.g., ERA5) and derived quantities needed by your architecture.  
4) **Split dataset** — generate a file list with `split` labels (`train/val/test`) pointing to processed parquet files.  
5) **Train** — use `ODETrainer` with `architecture_name` set to your target architecture and `model_params` from its `model.py`.  
6) **Infer** — load checkpoints with `NodeFDMPredictor`, reuse the architecture’s preprocessing hook, and write predictions.  
7) **Evaluate/visualize** — compute metrics and plot overlays for the columns your architecture outputs.

## OpenSky 2025 (ADS-B) example pipeline
1) **Aircraft sampling** — `code/opensky/01_aircraft_list.py` builds `data/aircraft_db.csv` from OpenSky Trino with `config.TYPECODES`.  
2) **Download raw data** — `code/opensky/02_download_data.py` pulls history, flightlist, and extended tables into `data/downloaded_parquet/`.  
3) **Decode & resample** — `code/opensky/03_preprocess_data.py <history.parquet>` decodes BDS 4/5/6, filters short flights, adds ADEP/ADES distance, resamples to 4s.  
4) **Weather + smoothing** — `code/opensky/04_weather_spd_process_data.py` adds ERA5 wind/temperature (`ERA5_FEATURES`), applies `selected_param_config`, writes `data/processed_flights/<typecode>/` and `dataset_split.csv`.  
5) **Train Neural ODEs** — `code/opensky/05_training.py` trains `opensky_2025` with `ODETrainer`, producing `models/opensky_<TYPECODE>/`.  
6) **Inference** — `code/opensky/06_flight_prediction.py` rolls out trajectories and stores `pred_*` columns.  
7) **Baselines and evaluation** — `07_bada_prediction.py` (if `BADA_4_2_DIR` set), `08_visualize_predictions.py` (overlays), `09_performance_aggregation.py` (MAE/MAPE/ME), `10_dataset_stats.py` (coverage).

## General tips
- Keep `config.py` as the single source of paths, typecodes, and architecture choices.  
- Remove hard-coded overrides (e.g., `acft = "A320"` in `05_training.py`) to loop over all types.  
- Remove the safety `break` in `04_weather_spd_process_data.py` if you want to process every preprocessed parquet (it currently stops after the first).  
- Ensure caches (e.g., `data/era5_cache`) exist to avoid repeated downloads.  
- For GPU/CPU constraints, lower `batch_size`, `num_workers`, and `seq_len` in `model_config`.
