from .PretrainedModel import * 
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, x):
        super().__init__()
        x = x.reshape((x.shape[0], *x.shape[1:]))
        n_in = x.shape[-1]
        self.net = nn.Linear(n_in, 10)

    def get_out_dim(self):
        return 10
    
    def forward(self, x):
        x = x.reshape((x.shape[0], *x.shape[1:]))
        return self.net(x)