from modules.preprocessing import SelectSplitData
import torch

class RejoinSplitData(SelectSplitData):
    def forward(self, x: torch.Tensor):
        """
        Does the reverse operation of what SelectSplitData does.
        """
        assert x.shape[0] % self.n_splits == 0, 'invalid shape of x!'
        return x.reshape((x.shape[0]//self.n_splits, *x.shape[1:-1], -1))