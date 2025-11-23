#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prediction helper to roll out flight trajectories with trained models."""

from typing import Dict, List

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from node_fdm.models.flight_dynamics_model_prod import FlightDynamicsModelProd


class NodeFDMPredictor:
    """Predict flight trajectories using a pretrained FlightDynamicsModelProd."""

    def __init__(
            self, 
            model_cols: list,
            model_path: Path, 
            dt: float = 4.0, 
            device: str = "cuda:0"
        ):
        """Initialize predictor with model path and column definitions.

        Args:
            model_cols: Sequence of model column groups (state, control, env, env_extra, derivatives).
            model_path: Directory containing pretrained model artifacts.
            dt: Integration timestep used for state propagation.
            device: Torch device string to run predictions on.
        """
        self.model_path = Path(model_path)
        self.x_cols, self.u_cols, self.e0_cols, self.e_cols, self.dx_cols = model_cols
        self.dt = dt
        self.device = torch.device(device)
        self.model = FlightDynamicsModelProd(model_path).to(self.device)
        self.model.eval()

    @staticmethod
    def _get_dict(f: pd.DataFrame, cols: List, i: int) -> Dict:
        """Slice a DataFrame row into a dict of tensors keyed by column definitions.

        Args:
            f: Flight data DataFrame.
            cols: Column identifiers to extract.
            i: Row index to slice.

        Returns:
            Dictionary mapping column identifiers to 1-sample tensors.
        """
        return {col: torch.tensor(f[col].iloc[i:i+1].values.astype(np.float32)) for col in cols}

    def _get_state(self, f: pd.DataFrame, i: int) -> Dict:
        """Extract state columns at a specific timestep.

        Returns:
            Dictionary mapping state columns to tensors.
        """
        return self._get_dict(f, self.x_cols, i)

    def _get_ctrl(self, f: pd.DataFrame, i: int) -> Dict:
        """Extract control columns at a specific timestep.

        Returns:
            Dictionary mapping control columns to tensors.
        """
        return self._get_dict(f, self.u_cols, i)

    def _get_env(self, f: pd.DataFrame, i: int) -> Dict:
        """Extract environment columns at a specific timestep.

        Returns:
            Dictionary mapping environment columns to tensors.
        """
        return self._get_dict(f, self.e0_cols, i)

    def _next_state(self, current_state: Dict, res_dict: Dict) -> Dict:
        """Advance state using predicted derivatives and configured timestep.

        Args:
            current_state: Mapping of current state tensors keyed by column.
            res_dict: Model output containing derivative tensors.

        Returns:
            Updated state mapping after one integration step.
        """
        new_state = dict()
        for x_col, (coeff, dx_col) in zip(self.x_cols, self.dx_cols):
            new_state[x_col] = current_state[x_col] + coeff * self.dt * res_dict[dx_col]
        return new_state

    def predict_flight(self, flight_df: pd.DataFrame, add_cols: list = []) -> pd.DataFrame:
        """Generate model predictions for an entire flight.

        Args:
            flight_df: Flight measurements DataFrame.
            add_cols: Optional extra columns to return alongside state predictions.

        Returns:
            DataFrame containing predicted columns with `pred_` prefix.
        """
        display_dict = {col: [] for col in self.x_cols + add_cols}

        current_state = self._get_state(flight_df, 0)
        current_state = {k: v.to(self.device) for k, v in current_state.items()}
        for i in range(len(flight_df)):
            input_dict = {**current_state, **self._get_ctrl(flight_df, i), **self._get_env(flight_df, i)}
            input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
            res_dict = self.model.forward(input_dict)
            for col in display_dict.keys():
                display_dict[col].append(res_dict[col].cpu().detach().numpy())
            current_state = self._next_state(current_state, res_dict)

        pred_df = pd.DataFrame({f"pred_{col}": np.concatenate(display_dict[col]) for col in display_dict}, index=flight_df.index)
        return pred_df
