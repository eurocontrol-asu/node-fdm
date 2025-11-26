# Create a new architecture

Follow these steps to add and register a new architecture (see the README guidance):

1) **Copy a skeleton**
- Duplicate `node_fdm/architectures/opensky_2025` (minimal) or `node_fdm/architectures/qar` (stacked layers) into `node_fdm/architectures/<your_arch>`.
- Keep the same file names: `columns.py`, `flight_process.py`, `model.py` (plus extra layers as needed).

2) **Declare columns** (`columns.py`)
- Define state (`X_COLS`), controls (`U_COLS`), environment (`E0_COLS`), derived outputs (`E1_COLS`), and derivatives (`DX_COLS`), using `utils.data.column.Column` and units.
- Make sure derivative columns match the ODE targets you want to learn.

3) **Custom preprocessing** (`flight_process.py`)
- Implement `flight_processing(df)` to add derived columns or smoothing.
- Optional `segment_filtering(df, start_idx, seq_len)` can reject bad segments (e.g., distance jumps).
- Expose any config you need (e.g., `selected_param_config`).

4) **Wire the model** (`model.py`)
- Build `X_COLS`, `U_COLS`, `E0_COLS`, `E1_COLS`, `DX_COLS`, and `MODEL_COLS`.
- Define layers (e.g., a physics/feature layer, then an ODE/data layer) and assemble `ARCHITECTURE` as a list of layer specs.
- Set `MODEL_COLS` to match the ordering expected by `FlightProcessor` and `SeqDataset`.

5) **Add any custom layers**
- Put them in the same folder (e.g., `trajectory_layer.py`, `engine_layer.py`).
- Export them via `__init__.py` if you need external imports.

6) **Register the name** (`architectures/mapping.py`)
- Add your key to `valid_names`.
- Ensure `get_architecture_module` imports your `columns`, `flight_process`, and `model`.

7) **Test a tiny run**
- Prepare a minimal processed dataset conforming to your columns.
- Run a short training with `ODETrainer` (small `seq_len`, small `batch_size`) to check shapes and stats.
- Verify inference with `NodeFDMPredictor` using your `MODEL_COLS` and `flight_processing`.

Tips:
- Keep column names consistent between preprocessing and model definitions.
- Use `Column` units to avoid silent scale bugs.
- Update the relevant pipeline `config.yaml` (paths, `typecodes`) if your dataset layout differs from the existing pipelines.
