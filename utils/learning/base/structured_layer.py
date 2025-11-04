
import torch
import torch.nn as nn
from utils.learning.base.blocks import Backbone, Head
from utils.learning.base.normalizers import InputNormalizer, OutputDenormalizer
from utils.learning.base.multi_layer_dict import MultiLayerDict


class StructuredLayer(nn.Module):
    def __init__(
        self,
        input_cols,
        input_stats,
        output_cols,
        output_stats,
        backbone_dim=48,
        backbone_depth=2,
        head_dim=24,
        head_depth=1,
    ):
        super().__init__()
        input_dim = len(input_cols)
        input_mean_dict, input_std_dict = input_stats
        output_mean_dict, output_std_dict, output_max_dict = output_stats

        self.input_cols = input_cols
        self.output_cols = output_cols

        self.normalizer = InputNormalizer(input_mean_dict, input_std_dict)
        self.backbone = Backbone(
            input_dim, hidden_dim=backbone_dim, num_layers=backbone_depth
        )

        def head_factory(col):
            return Head(
                backbone_dim,
                hidden_dim=head_dim,
                output_dim=1,
                num_layers=head_depth,
                last_activation=col.last_activation_fn,
            )

        self.heads = MultiLayerDict(self.output_cols, head_factory)

        self.denormalizer = OutputDenormalizer(
            output_mean_dict, output_std_dict, output_max_dict
        )

    def normalize_input(self, x_dict):
        out_list = []
        for col in self.input_cols:
            norm_vect = self.normalizer(x_dict[col], col)
            if len(norm_vect.shape) == 1:
                norm_vect = norm_vect.unsqueeze(1)
            out_list.append(norm_vect)
        return out_list

    def denormalize_output(self, out_norm_dict):
        out_pred_dict = dict()
        for col in self.output_cols:
            out_norm = out_norm_dict[col]
            out_pred_dict[col] = self.denormalizer(out_norm.squeeze(-1), col)
        return out_pred_dict

    def forward_trunk_head(self, x_dict):
        out_list = self.normalize_input(x_dict)
        x_norm = torch.cat(out_list, dim=1)
        features = self.backbone(x_norm)
        out_norm_dict = self.heads(features)
        return out_norm_dict

    def forward(self, x_dict):
        out_norm_dict = self.forward_trunk_head(x_dict)
        out_pred_dict = self.denormalize_output(out_norm_dict)
        return out_pred_dict
