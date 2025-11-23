from node_fdm.architectures.qar.columns import (
    col_n1,
    col_ff,
)
    
from utils.learning.base.structured_layer import StructuredLayer

class FuelN1Layer(StructuredLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n1_max = 100


    def forward(self, x_dict):
        out_norm_dict = self.forward_trunk_head(x_dict)
        
        fuel_flow_norm = out_norm_dict[col_ff].squeeze(-1)
        fuel_flow_pred = self.denormalizer(fuel_flow_norm, col_ff)
        
        
        out_pred_dict = dict()
        out_pred_dict[col_n1] = self.n1_max * out_norm_dict[col_n1].squeeze(-1)
        out_pred_dict[col_ff] = fuel_flow_pred
        
        return out_pred_dict