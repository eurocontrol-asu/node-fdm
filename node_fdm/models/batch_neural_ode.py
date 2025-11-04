import torch
import torch.nn as nn

class BatchNeuralODE(nn.Module):
    def __init__(self, model, u_seq, e_seq, t_grid):
        super().__init__()
        self.model = model
        self.model.reset_history()
        self.u_seq = u_seq
        self.e_seq = e_seq
        self.t_grid = t_grid

    def forward(self, t, x):
        t = t.item()
        idx = torch.searchsorted(self.t_grid, torch.tensor(t, device=self.t_grid.device)).item()
        idx0 = max(0, idx - 1)
        idx1 = min(idx, self.t_grid.shape[0] - 1)

        t0, t1 = self.t_grid[idx0].item(), self.t_grid[idx1].item()
        alpha = 0 if t1 == t0 else (t - t0) / (t1 - t0)

        u0, u1 = self.u_seq[:, idx0, :], self.u_seq[:, idx1, :]
        e0, e1 = self.e_seq[:, idx0, :], self.e_seq[:, idx1, :]

        u_t = (1 - alpha) * u0 + alpha * u1
        e_t = (1 - alpha) * e0 + alpha * e1

        return self.model(x, u_t, e_t)