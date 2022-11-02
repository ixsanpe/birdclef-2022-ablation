import torch.nn as nn

class OnlyXTransform(nn.Module):
    """
    Depreciated Module that would select only the first element of a tuple. 
    We are now using dicts instead of tuples and this code is not used any more 
    but kept if we would like to revert changes
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, tuple):
            return x[0] 
        else:
            return x 