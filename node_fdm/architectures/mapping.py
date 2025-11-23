import json

from node_fdm.architectures.opensky_2025 import (
    columns as opensky_2025_columns, 
    flight_process as opensky_2025_flight_process, 
    model as opensky_2025_model
)


from node_fdm.architectures.qar import (
    columns as qar_columns, 
    flight_process as qar_flight_process, 
    model as qar_model
)


def get_architecture_module(target_name: str):
    """
    Retrieve the architecture module based on the target name.
    """
    architecture_modules = {
        "opensky_2025": {
            "columns": opensky_2025_columns,
            "custom_fn": (
                opensky_2025_flight_process.flight_processing,
                opensky_2025_flight_process.segment_filtering,
                ),
            "model": opensky_2025_model,
        },
        "qar": {
            "columns": qar_columns,
            "custom_fn": (
                qar_flight_process.flight_processing,
                qar_flight_process.segment_filtering,
                ),
            "model": qar_model,
        },
        # Add other architectures here as needed
    }
    
    if target_name in architecture_modules:
        return architecture_modules[target_name]
    else:
        raise ValueError(f"Architecture '{target_name}' not found.")
    


def get_architecture_from_name(architecture_name):
    architecture_dict = get_architecture_module(architecture_name)
    architecture = architecture_dict["model"].ARCHITECTURE
    model_cols = architecture_dict["model"].MODEL_COLS
    custom_fn = architecture_dict["custom_fn"]
    return architecture, model_cols, custom_fn


def get_architecture_params_from_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    architecture, model_cols, _ = get_architecture_from_name(meta["architecture_name"])
    all_cols_dict = {str(col) : col for cols in model_cols for col in cols}
    meta_dict = meta["stats_dict"]
    stats_dict = {}
    for str_col, stats in meta_dict.items():
        col = all_cols_dict[str_col]
        stats_dict[col] =stats
    return architecture, model_cols, meta["model_params"], stats_dict