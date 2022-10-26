import torch
import torch.nn as nn


class InstanceNorm(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if isinstance(x, tuple):
            return self.forward_x(x[0]), x[1]
        else:
            return self.forward_x(x)

    def forward_x(self, x: torch.Tensor):
        """
        Select for a duration and split the data. If the duration is too short, we pad with 0's
        Parameters:
            x:
                array or tensor from which to select and to split
        Returns:
            processed version of x with shape (n_splits, ..., x.shape[-1]//n_folds)
        """
        instance_norm = nn.InstanceNorm2d(1)
        return instance_norm(x)
