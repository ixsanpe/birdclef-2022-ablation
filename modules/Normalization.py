import torch
import torch.nn as nn


class InstanceNorm(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: dict):
        """
        Select for a duration and split the data. If the duration is too short, we pad with 0's
        Parameters:
            x:
                array or tensor from which to select and to split
        Returns:
            processed version of x with shape (n_splits, ..., x.shape[-1]//n_folds)
        """
        instance_norm = nn.InstanceNorm2d(1)
        x['x'] = instance_norm(x['x'])
        return x
