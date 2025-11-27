# 🏗️ Create a New Architecture

Follow these steps to add and register a new architecture to the framework (see README guidance).

---

## 1. Copy a Skeleton

Start by duplicating an existing architecture folder into `node_fdm/architectures/<your_arch>`.

* **Minimal start**: Copy `node_fdm/architectures/opensky_2025`.
* **Advanced stack**: Copy `node_fdm/architectures/qar` (includes stacked layers).

**Required Files:**
* `columns.py`
* `flight_process.py`
* `model.py`
* `__init__.py` (plus any extra layer files).

---

## 2. Declare Columns (`columns.py`)

Define the variable groups using `utils.data.column.Column`. Ensure you specify units to avoid silent scale bugs.

* **`X_COLS`**: State variables (ODE inputs/outputs).
* **`U_COLS`**: Control variables.
* **`E0_COLS`**: Environmental inputs.
* **`E1_COLS`**: Derived outputs.
* **`DX_COLS`**: Derivatives predicted by the ODE layer.

!!! warning "Target Matching"
    Make sure your derivative columns (`DX_COLS`) exactly match the ODE targets you want the model to learn.

---

## 3. Custom Preprocessing (`flight_process.py`)

Implement the data preparation logic.

* **`flight_processing(df)`**: Augment raw data (e.g., add derived columns, apply smoothing). Expose specific configs here (e.g., `selected_param_config`).
* **`segment_filtering(df, start_idx, seq_len)`** *(Optional)*: Reject bad segments (e.g., segments with large distance jumps) before they reach the trainer.

---

## 4. Wire the Model (`model.py`)

This is where you define the architecture stack.

1.  **Build Column Lists**: Build `X_COLS`, `U_COLS`, `E0_COLS`, `E1_COLS`, `DX_COLS`, and `MODEL_COLS`.
2.  **Define Layers**: Define layers (e.g., a physics/feature layer, then an ODE/data layer) and assemble `ARCHITECTURE` as a list of layer specs.

!!! danger "Column Ordering"
    Set `MODEL_COLS` carefully. It must match the ordering expected by `FlightProcessor` and `SeqDataset`.

---

## 5. Add Custom Layers

If your model requires specific physics or feature logic:
1.  Place the files in your architecture folder (e.g., `trajectory_layer.py`, `engine_layer.py`).
2.  Export them via `__init__.py` if external imports are needed.

---

## 6. Register the Name

To make your architecture discoverable, edit `architectures/mapping.py`.

1.  Add your unique key to `valid_names`.
2.  Ensure `get_architecture_module` correctly imports your `columns`, `flight_process`, and `model` modules.

---

## 7. Test a Tiny Run

Before launching a full training job, verify the integration:

* **Data preparation**: Prepare a minimal processed dataset conforming to your new columns.
* **Training loop**: Run a short training with `ODETrainer` (using small `seq_len` and `batch_size`) to check tensor shapes and statistics.
* **Inference flow**: Verify `NodeFDMPredictor` using your `MODEL_COLS` and `flight_processing`.

---

## 💡 Pro Tips

!!! tip "Best Practices"
    * **Consistency**: Keep column names consistent between preprocessing and model definitions.
    * **Units**: Use `Column` units definition strictly to prevent silent scale bugs.
    * **Config**: Update the relevant pipeline `config.yaml` (paths, `typecodes`) if your dataset layout differs from the standard pipelines.

---

## 🚀 Next Steps

* **[Train a Model](../train_model/)**: Now that your architecture is registered, train it.