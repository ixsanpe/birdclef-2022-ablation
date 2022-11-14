import torch
from torch import nn

class SelectSplitData(nn.Module):
    def __init__(
        self,
        duration: int,
        n_splits: int,
        offset=None,
        sr=16000
    ):

        """
        Select a chunk of size duration and split it into n_splits
        Parameters:
            duration:
                time in seconds to extract from data
            n_splits:
                number of splits to make
            offset:
                when to start loading. If None, we choose a random offset each time
            sr:
                sampling rate of loaded data
        """
        super().__init__()
        self.duration = duration
        self.n_splits = n_splits
        self.offset = offset
        self.sr = sr

    def handle_shape(self, x: torch.Tensor):
        """
        pad x to compatible length with respect to self.duration, self.sr
        """
        if x.shape[-1] < self.sr * self.duration:
            missing = torch.zeros((*x.shape[:-1], self.sr * self.duration - x.shape[-1]))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.concat([x, missing], axis=-1).to(device)
        return x