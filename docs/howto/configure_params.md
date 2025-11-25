# Configure project paths and options

All runtime settings live in `config.py` (repository root):

- `DATA_DIR` and subfolders: where raw, preprocessed, processed, predicted, and BADA flights are stored.
- `MODELS_DIR`: checkpoints produced by `ODETrainer`.
- `ERA5_CACHE_DIR` and `ERA5_FEATURES`: meteorological cache and variables used in enrichment.
- `TYPECODES`: aircraft types processed by the OpenSky pipeline.
- `BADA_4_2_DIR`: **must be set** if you want to run the BADA baseline.
- `DEFAULT_CPU_COUNT`: used by preprocessing and parallel baselines.

Example:
```python
from config import DATA_DIR, MODELS_DIR, TYPECODES, ERA5_FEATURES

print("Data root:", DATA_DIR)
print("Model dir:", MODELS_DIR)
print("Types:", TYPECODES)
print("ERA5 features:", ERA5_FEATURES)
```

Recommendations:
- Keep paths relative to the repository root to avoid surprises when running scripts.
- Update `TYPECODES` once in `config.py` instead of editing each script.
- Ensure the directories exist before running downloads/preprocessing (`mkdir -p ...`).
- Set `BADA_4_2_DIR` to the directory containing the BADA 4.2 data files if you plan to run `07_bada_prediction.py`.
