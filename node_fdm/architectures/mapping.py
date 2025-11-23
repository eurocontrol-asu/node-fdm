import json
import importlib


def get_architecture_module(name: str):
    """
    Dynamically import only the architecture requested.
    """

    valid_names = ["opensky_2025", "qar"]

    if name not in valid_names:
        raise ValueError(f"Unknown architecture '{name}'. Valid names: {valid_names}")

    # dynamic import: node_fdm.architectures.<name>
    module_root = f"node_fdm.architectures.{name}"

    # submodules imported only when necessary
    columns = importlib.import_module(f"{module_root}.columns")
    flight_process = importlib.import_module(f"{module_root}.flight_process")
    model = importlib.import_module(f"{module_root}.model")

    return {
        "columns": columns,
        "custom_fn": (
            flight_process.flight_processing,
            flight_process.segment_filtering,
        ),
        "model": model,
    }



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
    x_cols, u_cols, e0_cols, e_cols, dx_cols = model_cols
    deriv_cols = [col.derivative for col in x_cols]
    model_cols2 = [x_cols, u_cols, e0_cols, e_cols, deriv_cols]

    # mapping string → column object
    all_cols_dict = {str(col): col for cols in model_cols2 for col in cols}
    stats_dict = {
        all_cols_dict[str_col]: stats
        for str_col, stats in meta["stats_dict"].items()
    }

    return architecture, model_cols, meta["model_params"], stats_dict
