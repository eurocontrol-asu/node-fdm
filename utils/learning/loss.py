import torch.nn as nn

        
def get_loss(loss_name):
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "bce":
        return nn.BCELoss()
    else:
        return nn.MSELoss()
