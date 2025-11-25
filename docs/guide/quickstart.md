# Quickstart (OpenSky 2025 pipeline)

This follows the scripts in `code/opensky` to reproduce the ADS-B workflow. Paths below assume `cd lib/python`.

## 0) Configure paths and types
- Edit `config.py` if you need different directories or aircraft typecodes (`TYPECODES`).
- Create data folders (see `Guide → Installation`).

## 1) Build the aircraft list
Generates `data/aircraft_db.csv` with sampled icao24 per type:
```bash
python code/opensky/01_aircraft_list.py
```
Requires OpenSky Trino access (see `traffic.data.opensky`).

## 2) Download OpenSky data
Downloads history, flightlist, and extended data for the sampled aircraft:
```bash
python code/opensky/02_download_data.py
```
By default it loops 2024-10-01 → 2025-10-15 in 20-day steps and stores parquet files under `data/downloaded_parquet/`.

## 3) Decode + resample flights
Decodes BDS 4/5/6, filters short flights, computes ADEP/ADES distances, resamples to 4s:
```bash
python code/opensky/03_preprocess_data.py data/downloaded_parquet/history_20241001.parquet --workers 8
```
One parquet per day is produced in `data/preprocessed_parquet`. Run once per downloaded history file.

## 4) Enrich with ERA5 and split
Adds ERA5 (wind, temperature), smooths selected parameters (`selected_param_config`), and builds `dataset_split.csv`:
```bash
python code/opensky/04_weather_spd_process_data.py
```
Outputs:
- Enriched flights in `data/processed_flights/<aircraft_type>/...`
- Global split file `data/processed_flights/dataset_split.csv`

## 5) Train Neural ODEs
```bash
python code/opensky/05_training.py
```
Current script trains the OpenSky 2025 architecture with `ODETrainer`; tweak `model_config` and the inner `acft = "A320"` line to run more types. Checkpoints land in `models/opensky_<TYPECODE>/`.

## 6) Run inference on the test split
```bash
python code/opensky/06_flight_prediction.py
```
Uses `NodeFDMPredictor` (device default: `cuda:0`) to produce `data/predicted_flights/<TYPECODE>/<flight>.parquet`.

## 7) BADA baseline (optional)
If `BADA_4_2_DIR` is configured and mappings exist:
```bash
python code/opensky/07_bada_prediction.py
```
Baseline predictions are saved to `data/bada_flights/<TYPECODE>/`.

## 8) Visualize and aggregate performance
- `code/opensky/08_visualize_predictions.py`: quick matplotlib overlays for truth vs. model vs. selections (and BADA if available).
- `code/opensky/09_performance_aggregation.py`: aggregates MAE/MAPE/ME per phase, saves `data/performance.parquet`.

For dataset coverage checks, see `code/opensky/10_dataset_stats.py`.
