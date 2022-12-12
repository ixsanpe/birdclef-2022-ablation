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
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            missing = torch.zeros((*x.shape[:-1], int(self.sr * self.duration - x.shape[-1]))).to(device)
            x = torch.concat([x, missing], axis=-1).to(device)
        return x

    def get_intervals(self, durations):
        # for each sample calculate the maximum allowed offset
        max_offset = (durations - self.duration * self.sr - 1).double()
        max_offset = torch.where(max_offset > 0., max_offset, 0.)

        # select an offset (randomly or self.offset)
        if self.offset is None:
            offset = torch.rand(durations.shape, device=max_offset.device) * max_offset
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            offset = torch.where(max_offset < self.offset, max_offset, self.offset).to(device)
        
        # select that data
        start = torch.ceil(offset).int()
        stop = start + self.duration * self.sr
        return start, stop.int() 
    
    def forward(self, d: dict):
        """
        Select for a duration and split the data. If the duration is too short, we pad with 0's
        Parameters:
            x:
                array or tensor from which to select and to split
            durations:
                the duration for each x so that we don't use many empty spectrograms
        Returns:
            processed version of x with shape (x.shape[0] * n_splits, ..., x.shape[-1]//n_folds)
        """
        # ensure input has at last the correct duration
        x = d['x']
        x = self.handle_shape(x)

        # select which data to pick
        durations = d['lens']
        start, stop = self.get_intervals(durations)
        # select that data
        waveform = torch.stack([x[idx, ..., i:j] for idx, (i, j) in enumerate(zip(start, stop))], axis=0)

        # return the reshaped data
        d['x'] = waveform.reshape((waveform.shape[0]*self.n_splits, *waveform.shape[1:-1], -1))
        return d
        
        
