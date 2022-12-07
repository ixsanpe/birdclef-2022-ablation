import torch.nn as nn 
import torch 

class SimpleAttention(nn.Module):
    """
    Example post-processing step

    Adapted from from Henkel et. al. 
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """
    def __init__(self, n_in, width=512, n_out=1):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, n_out))
    
    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(1)
        weight = torch.softmax(self.att(x), axis=1)
        output = (x * weight)
        return output.sum(1)