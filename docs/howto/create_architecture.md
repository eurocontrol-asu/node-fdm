# 🏗️ Create a New Architecture

Follow these steps to add and register a new architecture to the framework.

---

## 1. Define Column Schemas

Create a new schema module in `node_fdm_data/schemas/`:

```python title="node_fdm_data/schemas/my_arch.py"
from __future__ import annotations

import polars as pl
from node_fdm_data.conversions import ft_to_m, kt_to_ms

X_COLS: list[str] = ["distance_m", "altitude_ft", "gamma_rad", "tas_kt"]
U_COLS: list[str] = ["alt_sel_ft", "mach_sel"]
E0_COLS: list[str] = ["long_wind_kt", "temperature_K"]
E1_COLS: list[str] = ["vz_ftmin", "mach", "cas_kt"]
DX_COLS: list[tuple[int, str]] = [(1, "gs_kt"), (1, "vz_ftmin")]

CONVERSIONS: dict[str, tuple] = {
    "altitude": (ft_to_m, "altitude_m"),
    "TAS": (kt_to_ms, "tas_ms"),
}
```

!!! warning "Target Matching"
    Make sure `DX_COLS` exactly match the derivatives you want the model to learn.

---

## 2. Implement Preprocessing

Create processing hooks in `node_fdm_data/preprocessing/`:

```python title="node_fdm_data/preprocessing/my_arch.py"
from __future__ import annotations

import polars as pl

def flight_processing(df: pl.LazyFrame) -> pl.LazyFrame:
    """Augment raw data before training."""
    return df.with_columns(...)

def segment_filtering(df: pl.DataFrame, start: int, seq_len: int) -> bool:
    """Reject bad segments before training."""
    return True
```

---

## 3. Register the Architecture

Create a spec in `node_fdm/architectures/`:

```python title="node_fdm/architectures/my_arch.py"
from __future__ import annotations

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm_data.schemas.my_arch import X_COLS, U_COLS, E0_COLS, E1_COLS, DX_COLS

MY_ARCH = ArchitectureSpec(
    name="my_arch",
    x_cols=X_COLS,
    u_cols=U_COLS,
    e0_cols=E0_COLS,
    e1_cols=E1_COLS,
    dx_cols=DX_COLS,
    layers=[
        LayerSpec(
            name="trajectory",
            layer_class="node_fdm.layers.trajectory.TrajectoryLayer",
            input_cols=X_COLS + U_COLS + E0_COLS,
            output_cols=E1_COLS,
            trainable=False,
        ),
        LayerSpec(
            name="structured",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=X_COLS + U_COLS + E0_COLS + E1_COLS,
            output_cols=[col for _, col in DX_COLS],
            trainable=True,
        ),
    ],
    preprocessing_fn="node_fdm_data.preprocessing.my_arch.flight_processing",
    segment_filter_fn="node_fdm_data.preprocessing.my_arch.segment_filtering",
)
register(MY_ARCH)
```

---

## 4. Add Custom Layers (Optional)

If your model requires specific physics logic:

1. Place layer files in `node_fdm/layers/` (e.g., `my_layer.py`)
2. Subclass `StructuredLayer` or `nn.Module`
3. Reference via dotted path in `LayerSpec.layer_class`

---

## 5. Test a Tiny Run

Before launching a full training job:

* **Data preparation**: Prepare a minimal dataset conforming to your columns
* **Training loop**: Run a short training with `ODETrainer` (small `seq_len` and `batch_size`)
* **Inference flow**: Verify `NodeFDMPredictor` loads and predicts correctly

---

## 💡 Pro Tips

!!! tip "Best Practices"
    * **Consistency**: Keep column names consistent between schemas, preprocessing, and architecture specs
    * **Frozen specs**: `ArchitectureSpec` is frozen (immutable) — define once, use everywhere
    * **Auto-registration**: Specs register on import — just make sure the module is imported

---

## 🚀 Next Steps

* **[Train a Model](../train_model/)**: Now that your architecture is registered, train it.
