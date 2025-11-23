#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers to dynamically load and assemble architecture components."""

import json
import importlib
from typing import Any, Dict, Tuple


def get_architecture_module(name: str) -> Dict[str, Any]:
    """Dynamically import only the architecture requested.

    Args:
        name: Architecture package name to load.

    Returns:
        Dictionary containing imported column, model, and custom function modules.

    Raises:
        ValueError: If the provided architecture name is not supported.
    """
    valid_names = ["opensky_2025", "qar"]

    if name not in valid_names:
        raise ValueError(f"Unknown architecture '{name}'. Valid names: {valid_names}")

    module_root = f"node_fdm.architectures.{name}"

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


def get_architecture_from_name(architecture_name: str) -> Tuple[Any, Any, Any]:
    """Return architecture definition, model columns, and custom functions by name.

    Args:
        architecture_name: Name of the architecture to load.

    Returns:
        Tuple of (architecture layers, model columns, custom functions).
    """
    architecture_dict = get_architecture_module(architecture_name)
    architecture = architecture_dict["model"].ARCHITECTURE
    model_cols = architecture_dict["model"].MODEL_COLS
    custom_fn = architecture_dict["custom_fn"]
    return architecture, model_cols, custom_fn


def get_architecture_params_from_meta(meta_path: str) -> Tuple[Any, Any, Any, Dict[Any, Any]]:
    """Load architecture parameters and stats from a meta JSON file.

    Args:
        meta_path: Path to the meta JSON file.

    Returns:
        Tuple containing architecture, model columns, model parameters, and stats dictionary.
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)

    architecture, model_cols, _ = get_architecture_from_name(meta["architecture_name"])
    x_cols, u_cols, e0_cols, e_cols, _ = model_cols
    deriv_cols = [col.derivative for col in x_cols]
    model_cols2 = [x_cols, u_cols, e0_cols, e_cols, deriv_cols]

    all_cols_dict = {str(col): col for cols in model_cols2 for col in cols}
    stats_dict = {
        all_cols_dict[str_col]: stats
        for str_col, stats in meta["stats_dict"].items()
    }

    return architecture, model_cols, meta["model_params"], stats_dict
