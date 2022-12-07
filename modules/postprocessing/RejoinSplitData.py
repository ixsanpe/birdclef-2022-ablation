from modules.preprocessing import SelectSplitData
import torch

class RejoinSplitData(SelectSplitData):
    """
    Does the reverse operation of what SelectSplitData does.
    """
    def forward(self, x: torch.Tensor): 
        assert x.shape[0] % self.n_splits == 0, f'invalid shape of x! {x.shape=}{self.n_splits=}'
        return x.reshape((x.shape[0]//self.n_splits, *x.shape[1:-1], -1))
        