import os
import torch
import torch.nn as nn
from utils.learning.base.structured_layer import StructuredLayer
from node_fdm.architectures.mapping import get_architecture_params_from_meta

class FlightDynamicsModelProd(nn.Module):
    def __init__(
        self,
        model_path,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        meta_path = model_path / "meta.json"
        self.architecture, self.model_cols, model_params, self.stats_dict = get_architecture_params_from_meta(meta_path)
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
            if name != "trajectory":
                checkpoint = self.load_layer_checkpoint(name)
                self.layers_dict[name].load_state_dict(
                    checkpoint["layer_state"], strict=False
                )
                self.layers_dict[name] = self.layers_dict[name].eval()

    def load_layer_checkpoint(self, layer_name):
        path = os.path.join(self.model_path, f"{layer_name}.pt")
        checkpoint = torch.load(path, map_location=self.device)
        return checkpoint

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

    def forward(self, vect_dict):
        for name in self.layers_name:
            vect_dict |= self.layers_dict[name](vect_dict)
        return vect_dict
