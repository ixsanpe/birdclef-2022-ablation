import torch.nn as nn
import torch

class ValidationSplit(nn.Module):
    def __init__(
            self,
            duration: int,
            n_chunks:int,
            n_splits: int,
            sr=16000
    ):

        '''
        Select chunks of size duration for every audio and apply a rolling windows.
        The number of chunks is fixed for every audio file: based on that we compute the rolling window
        Parameters:
            duration:
                time in seconds to extract from data
            n_chunks:
                number of times to extract segments of size duration
            n_splits:
                number of splits to make on every chunk
        '''
