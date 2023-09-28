from torch import nn
import torch

class FrechetModule(nn.Module):
    '''
    Parent for a module that can be used the calculate the FID, using this module instead of the inception net.
    Torchmetric requires the module to output a vector (batch_size, channels)
    '''
    def set_as_fid_model(self):
        self.used_for_fid = True

class NormToZeroOne(nn.Module):
    def forward(self, x):
        min = torch.min(x)
        max = torch.max(x)

        return (x - min)/(max - min)

class IdentityModule(nn.Module):
    def forward(self, x):
        return x
    
class DebugLogger(nn.Module):
    def __init__(self, prefix="") -> None:
        super(DebugLogger, self).__init__()
        self.prefix = prefix

    def forward(self, x):
        print(f"----{self.prefix}----")
        print(x.shape)
        # print(f"---/{self.prefix}----")
        return x

def freeze_parameters(model: nn.Module, requires_grad=False):
    for param in model.parameters():
        param.requires_grad_(requires_grad)
    return model
