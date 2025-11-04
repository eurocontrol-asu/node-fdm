import torch.nn as nn


class MultiLayerDict(nn.Module):
    def __init__(self, output_cols, layer_factory):
        super().__init__()
        self.output_cols = output_cols
        col_dict = {
            col.col_name : layer_factory(col)
            for col in self.output_cols}
        self.layer_dict = nn.ModuleDict(col_dict)
        
    def forward(self, x):
        output_dict = {
            col: self.layer_dict[col.col_name](x) 
            for col in self.output_cols
        }
        return output_dict