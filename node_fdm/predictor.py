import torch
import pandas as pd
import numpy as np
from pathlib import Path
from node_fdm.models.flight_dynamics_model_prod import FlightDynamicsModelProd

class NodeFDMPredictor:
    def __init__(
            self, 
            model_cols: list,
            model_path: Path, 
            dt: float = 4.0, 
            device: str = "cuda:0"
        ):
        self.model_path = Path(model_path)
        self.x_cols, self.u_cols, self.e0_cols, self.e_cols, self.dx_cols = model_cols
        self.dt = dt
        self.device = torch.device(device)
        self.model = FlightDynamicsModelProd(model_path).to(self.device)
        self.model.eval()

    # --- Dict builders ---
    @staticmethod
    def _get_dict(f, cols, i):
        return {col: torch.tensor(f[col].iloc[i:i+1].values.astype(np.float32)) for col in cols}

    def _get_state(self, f, i): return self._get_dict(f, self.x_cols, i)
    def _get_ctrl(self, f, i): return self._get_dict(f, self.u_cols, i)
    def _get_env(self, f, i):  return self._get_dict(f, self.e0_cols + self.e_cols, i)

    # --- State propagation ---
    def _next_state(self, current_state, res_dict):
        new_state = dict()
        for x_col, (coeff, dx_col) in zip(self.x_cols, self.dx_cols):
            new_state[x_col] = current_state[x_col] + coeff * self.dt * res_dict[dx_col]
        return new_state

    # --- Predict a single flight ---
    def predict_flight(self, flight_df: pd.DataFrame, add_cols: list = []) -> pd.DataFrame:
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