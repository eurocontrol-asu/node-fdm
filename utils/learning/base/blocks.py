import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=1, 
                 last_activation=None):
        
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        if last_activation is not None:
            layers.append(last_activation())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class Backbone(MLPBlock):
    def __init__(self, input_dim, hidden_dim=48, num_layers=2, last_activation=None):
        super().__init__(
            input_dim, 
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            last_activation=last_activation
        )

class Head(MLPBlock):
    def __init__(self, input_dim, hidden_dim=24, output_dim=1, num_layers=1, last_activation=None):
        super().__init__(
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_layers=num_layers,
            last_activation=last_activation
        )