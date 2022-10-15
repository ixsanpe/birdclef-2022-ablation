import torch.nn as nn 
import torch 

class SimpleAttention(nn.Module):
    """
    Example post-processing step
    """
    def __init__(self, n_in, width=512, n_out=1):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, n_out))
    
    def forward(self, x):
        weight = torch.softmax(self.att(x), axis=1)
        return (x * weight).sum(1) 