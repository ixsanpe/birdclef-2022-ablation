import torch
import torch.nn as nn


class InstanceNorm(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: dict):
        """
        apply instance normalization to the data contained in the input dictionary x 
        (i.e. x['x'])
        """
        instance_norm = nn.InstanceNorm2d(1)
        x['x'] = instance_norm(x['x'])
        return x
