import torch
import pandas as pd
import numpy as np
from pathlib import Path
from node_fdm.models.flight_dynamics_model_prod import FlightDynamicsModelProd

from node_fdm.architectures.opensky_2025.model import X_COLS, U_COLS, E0_COLS, E1_COLS
from node_fdm.architectures.opensky_2025.columns import col_dist, col_alt, col_gamma, col_tas, col_vz, col_cas, col_mach, col_gs

class NodeFDMPredictor:
    def __init__(self, model_path: Path, dt: float = 4.0, device: str = "cuda:0"):
        self.model_path = Path(model_path)
        self.dt = dt
        self.device = torch.device(device)
        self.model = FlightDynamicsModelProd(model_path).to(self.device)
        self.model.eval()

    # --- Dict builders ---
    @staticmethod
    def _get_dict(f, cols, i):
        return {col: torch.tensor(f[col].iloc[i:i+1].values.astype(np.float32)) for col in cols}

    def _get_state(self, f, i): return self._get_dict(f, X_COLS, i)
    def _get_ctrl(self, f, i): return self._get_dict(f, U_COLS, i)
    def _get_env(self, f, i):  return self._get_dict(f, E0_COLS + E1_COLS, i)

    # --- State propagation ---
    def _next_state(self, current_state, res_dict):
        new_state = dict()
        new_state[col_dist] = current_state[col_dist] + self.dt * res_dict[col_gs]
        new_state[col_alt]  = current_state[col_alt]  + self.dt * res_dict[col_vz]
        new_state[col_gamma] = current_state[col_gamma] + self.dt * res_dict[col_gamma.derivative]
        new_state[col_tas] = current_state[col_tas] + self.dt * res_dict[col_tas.derivative]
        return new_state

    # --- Predict a single flight ---
    def predict_flight(self, flight_df: pd.DataFrame):
        add_cols = [col_cas, col_tas.derivative, col_mach, col_vz]
        display_dict = {col: [] for col in X_COLS + add_cols}

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