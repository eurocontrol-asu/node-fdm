import torch
import torch.nn as nn

from node_fdm.architectures.opensky_2025.columns import (
    col_gs,
    col_vz,
    col_gamma,
    col_tas,
)
from utils.learning.base.structured_layer import StructuredLayer


class FlightDynamicsModel(nn.Module):
    def __init__(
        self,
        architecture,
        stats_dict,
        model_cols,
        model_params=[2, 1, 48],
    ):
        super().__init__()
        self.architecture = architecture
        self.stats_dict = stats_dict
        self.x_cols, self.u_cols, self.e0_cols, self.e_cols, self.dx_cols = model_cols
        self.backbone_depth, self.head_depth, self.neurons_num = model_params
        self.layers_dict = nn.ModuleDict({})
        self.layers_name = []

        for name, layer_class, input_cols, ouput_cols, structured in self.architecture:
            self.layers_name.append(name)
            if structured:
                self.layers_dict[name] = self.create_structured_layer(
                    input_cols,
                    ouput_cols,
                    layer_class=layer_class,
                )
            else:
                self.layers_dict[name] = layer_class()

    def reset_history(self):
        self.history = {}

    def create_structured_layer(
        self,
        input_cols,
        output_cols,
        layer_class=StructuredLayer,
    ):
        input_stats = [
            {
                col.col_name: self.stats_dict[col][metric]
                for col in input_cols
                if col.normalize_mode is not None
            }
            for metric in ["mean", "std"]
        ]
        output_stats = [
            {
                col.col_name: self.stats_dict[col][metric]
                for col in output_cols
                if col.denormalize_mode is not None
            }
            for metric in ["mean", "std", "max"]
        ]

        layer = layer_class(
            input_cols,
            input_stats,
            output_cols,
            output_stats,
            backbone_dim=self.neurons_num,
            backbone_depth=self.backbone_depth,
            head_dim=self.neurons_num // 2,
            head_depth=self.head_depth,
        )

        return layer

    def forward(self, x, u_t, e_t):

        vects = torch.cat([x, u_t, e_t], dim=1)
        vect_dict = dict()
        for i, col in enumerate(self.x_cols + self.u_cols + self.e0_cols + self.e_cols):
            vect_dict[col] = vects[..., i]

        for name in self.layers_name:
            vect_dict = vect_dict | self.layers_dict[name](vect_dict)

        ode_output = torch.stack(
            [
                vect_dict[col_gs],
                vect_dict[col_vz],
                vect_dict[col_gamma.derivative],
                vect_dict[col_tas.derivative],
            ],
            dim=1,
        )

        for col, vect in vect_dict.items():
            if torch.isnan(vect).any():
                pass
            if col in self.history.keys():
                self.history[col] = torch.cat(
                    [self.history[col], vect.unsqueeze(1)], dim=1
                )
            else:
                self.history[col] = vect.unsqueeze(1)

        return ode_output
