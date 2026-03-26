"""Input normalization and output denormalization for structured layers.

Column-aware normalization uses plain string column names with a ``modes``
dictionary mapping each column to its normalization strategy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "InputNormalizer",
    "OutputDenormalizer",
]


class InputNormalizer(nn.Module):
    """Normalize inputs based on per-column mean/std statistics.

    Each column can be either ``"normal"`` (z-score) or ``None`` (passthrough).
    """

    def __init__(
        self,
        mean_dict: dict[str, float],
        std_dict: dict[str, float],
        modes: dict[str, str | None] | None = None,
    ) -> None:
        """Register mean and std buffers for each column.

        Args:
            mean_dict: Mapping column name → mean value.
            std_dict: Mapping column name → std value.
            modes: Mapping column name → normalize mode (``"normal"`` or ``None``).
                If ``None``, all columns with stats are normalized.
        """
        super().__init__()
        self._modes = modes or dict.fromkeys(mean_dict, "normal")
        for k in mean_dict:
            self.register_buffer(f"mean_{k}", torch.tensor(mean_dict[k], dtype=torch.float32))
            self.register_buffer(f"std_{k}", torch.tensor(std_dict[k], dtype=torch.float32))

    def forward(self, x: torch.Tensor, col: str) -> torch.Tensor:
        """Normalize input tensor for the given column.

        Args:
            x: Input tensor.
            col: Column name.

        Returns:
            Normalized tensor or original if mode is ``None``.
        """
        if self._modes.get(col) == "normal":
            mean: torch.Tensor = getattr(self, f"mean_{col}")
            std: torch.Tensor = getattr(self, f"std_{col}")
            return (x - mean) / std
        return x


class OutputDenormalizer(nn.Module):
    """Denormalize network outputs to physical scale.

    Supports ``"normal_clamp"`` (z-score + clamp) and ``"max"`` (scale by max).
    """

    def __init__(
        self,
        mean_dict: dict[str, float],
        std_dict: dict[str, float],
        max_dict: dict[str, float],
        modes: dict[str, str | None] | None = None,
        max_ratio: float = 1.2,
        scale_dict: dict[str, float] | None = None,
        cap_dict: dict[str, float] | None = None,
    ) -> None:
        """Register normalization statistics per column.

        Args:
            mean_dict: Mapping column name → mean.
            std_dict: Mapping column name → std.
            max_dict: Mapping column name → max absolute value.
            modes: Mapping column name → denormalize mode
                (``"normal_clamp"``, ``"max"``, ``"scaled"``, or ``None``).
                Defaults to ``"normal_clamp"`` for all columns.
            max_ratio: Clamping ratio applied to max values.
            scale_dict: Mapping column name → scale for ``"scaled"`` mode.
            cap_dict: Mapping column name → cap for ``"scaled"`` mode.
        """
        super().__init__()
        self.max_ratio = max_ratio
        self._modes = modes or dict.fromkeys(mean_dict, "normal_clamp")
        for k in mean_dict:
            self.register_buffer(f"mean_{k}", torch.tensor(mean_dict[k], dtype=torch.float32))
            self.register_buffer(f"std_{k}", torch.tensor(std_dict[k], dtype=torch.float32))
            self.register_buffer(f"max_{k}", torch.tensor(max_dict[k], dtype=torch.float32))
        _scale = scale_dict or {}
        _cap = cap_dict or {}
        for k in _scale:
            self.register_buffer(f"scale_{k}", torch.tensor(_scale[k], dtype=torch.float32))
        for k in _cap:
            self.register_buffer(f"cap_{k}", torch.tensor(_cap[k], dtype=torch.float32))

    def forward(self, x: torch.Tensor, col: str) -> torch.Tensor:
        """Denormalize output tensor for the given column.

        Args:
            x: Normalized tensor.
            col: Column name.

        Returns:
            Tensor in physical scale.
        """
        mode = self._modes.get(col)
        if mode == "normal_clamp":
            mean: torch.Tensor = getattr(self, f"mean_{col}")
            std: torch.Tensor = getattr(self, f"std_{col}")
            maxv: torch.Tensor = getattr(self, f"max_{col}")
            value = mean + x * std
            return torch.clamp(value, min=-self.max_ratio * maxv, max=self.max_ratio * maxv)
        if mode == "scaled":
            scale: torch.Tensor = getattr(self, f"scale_{col}")
            cap: torch.Tensor = getattr(self, f"cap_{col}")
            return torch.clamp(x * scale, min=-cap, max=cap)
        if mode == "max":
            maxv_val: torch.Tensor = getattr(self, f"max_{col}")
            return x * maxv_val
        return x
