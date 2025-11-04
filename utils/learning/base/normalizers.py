import torch 
import torch.nn as nn

class InputNormalizer(nn.Module):
    def __init__(self, mean_dict, std_dict):
        super().__init__()
        for k in mean_dict:
            k = str(k)
            self.register_buffer(f'mean_{k}', torch.tensor(mean_dict[k], dtype=torch.float32))
            self.register_buffer(f'std_{k}', torch.tensor(std_dict[k], dtype=torch.float32))
    
    def forward(self, x, col):
        if col.normalize_mode == "normal":
            mean = getattr(self, f'mean_{col}')
            std = getattr(self, f'std_{col}')
            return (x - mean) / std
        return x


class OutputDenormalizer(nn.Module):
    def __init__(self, mean_dict, std_dict, max_dict, max_ratio=1.2):
        super().__init__()
        self.max_ratio = max_ratio
        for k in mean_dict:
            k = str(k)
            self.register_buffer(f'mean_{k}', torch.tensor(mean_dict[k], dtype=torch.float32))
            self.register_buffer(f'std_{k}', torch.tensor(std_dict[k], dtype=torch.float32))
            self.register_buffer(f'max_{k}', torch.tensor(max_dict[k], dtype=torch.float32))

    def forward(self, x, col):
        if col.denormalize_mode == "normal_clamp":
            mean = getattr(self, f'mean_{col}')
            std = getattr(self, f'std_{col}')
            maxv = getattr(self, f'max_{col}')
            value = mean + x * std
            value_clamped = torch.clamp(value, min=-self.max_ratio * maxv, max=self.max_ratio * maxv)
            return value_clamped
        elif col.denormalize_mode == "max":
            maxv = getattr(self, f'max_{col}')
            value = x * maxv
            return value
        return x

