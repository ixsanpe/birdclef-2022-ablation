from typing import Optional
import torch

class Selector():
    def __init__(self, duration: int, offset: Optional[int], device: Optional[str]) -> None:
        """
        Class that selects only a portion of a single tensor
        """
        self.duration = duration 
        self.offset = offset 
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
    
    def __call__(self, x):
        """
        Select self.duration elements of the final
        axis of x. If self.offset is None, we pick
        randomly from the interval [0, x.shape[-1] - self.duration]
        """
        dur_x = x.shape[-1]
        max_offset = max([0, dur_x - self.duration])
        if self.offset is None:
            offset = torch.rand(size=(1, ), device=self.device) * max_offset
        else:
            offset = min([max_offset, self.offset])
        return x[..., int(offset): int(offset) + self.duration]