import torch.nn as nn

class OnlyXTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, tuple):
            return x[0]
        else:
            return x 