import torch.nn as nn 
import torch 
from modules import SelectSplitData, TransformApplier
from typing import Callable
import warnings
from math import ceil
from copy import deepcopy

class Validator():
    def __init__(
            self, 
            data_pipeline, 
            model, 
            device: str, 
            overlap: float=.5, 
            bs_max=32,
            policy: Callable=lambda l: torch.stack(l, axis=-1).max(axis=-1).values
        ):
        """
        Implements a validate function. We use a separate class for this to handle overlapping
        windows in the spectrogram
        Parameters:
            data_pipeline:
                how to transform the input data
            model:
                model to make predictions
            overlap:
                how much the windows should overlap when computing the predictions
            bs_max:
                when validating, what is the maximum allowed batch size
            policy:
                A function that defines how to compute the final predictions from the predictions
                of the individual segments
        """
        super().__init__()
        self.data_pipeline = data_pipeline
        self.model = model
        assert overlap < 1 and overlap >= 0, f'overlap has to be in [0, 1) but got {overlap=}'
        self.overlap = overlap
        self.policy = policy
        self.device = device
        self.bs_max = bs_max

        # check for splitting data
        self.data_splitter = None 
        if not self.find_data_splitter():
            warnings.warn('Warning, found no SelectSplitData in Validator')

    def find_data_splitter(self, applier=None):
        """
        Recursively check self.datapipeline and any TransformAppliers therein
        for a SelectSplitData instance
        """
        to_check = applier if applier is not None else self.data_pipeline
        if any([isinstance(e, SelectSplitData) for e in list(to_check)]):
            data_splitters = [e for e in list(to_check) if isinstance(e, SelectSplitData)]
            assert len(data_splitters) <= 1, f'validation only implemented for one datasplitter but found {len(data_splitters)}'
            self.data_splitter = None if len(data_splitters) == 0 else data_splitters[0]
            return True

        transform_appliers = [e for e in list(to_check) if isinstance(e, TransformApplier)]
        return any([self.find_data_splitter(applier) for applier in transform_appliers])
    
    def predict_rolling(self, d_copy, skip, N_segments, n_duration, offset=0):
        
        copies = [torch.roll(d_copy['x'], shifts=(offset+i)*skip, dims=-1)[..., :ceil(n_duration)] for i in range(N_segments)]
        
        """
        make a tensor like 
        [
          batch1 shift1, 
          batch2 shift1, 
          ..., 
          batchn shift1, 
          
          batch1 shift2, 
          ..., 
          batchn shift2, 
             
          ...
          
          batch1 shiftN, ...
        ]
        """
        
        copies = torch.concat(copies, axis=0)
        
        """
        To facilitate quicker execution, we pretend that each offset audio file is its own "batch"

        the batch size here does not exceed self.bs_max
        """
        for k, v in d_copy.items():
            if k == 'x':
                d_copy['x'] = copies 
            else:
                if isinstance(v, torch.Tensor):
                    d_copy[k] = torch.concat([v]*N_segments, axis=0)
                elif isinstance(v, list):
                    d_copy[k] = [v]*N_segments
        logits = self.forward_item(d_copy)[0] # preds for (b1s1, b2s1, ..., b1s2, ..., b1sN, ...)
        return logits 

    def predict(self, bs, d_copy, skip, N_segments, n_duration):
        outputs = []
        # for this operation we require divisibility
        if self.bs_max % bs != 0:
            self.bs_max = self.bs_max - self.bs_max % bs

        N_iterations = ceil(bs*N_segments / self.bs_max)
        for i in range(N_iterations):
            outputs.append(self.predict_rolling(
                d_copy.copy(), 
                skip, 
                N_segments=min( # limit output to max_bs. But make sure we don't take too much using the max
                    [(self.bs_max//bs), max([N_segments - i*(self.bs_max//bs), 0])]
                ), 
                n_duration=n_duration, 
                offset=i*self.bs_max
            ))
        # now outputs is a list of lists. outputs[i] = list of length self.bs_max//bs, containing
        # the predictions for the i to i+1-th chunks of size self.bs_max//bs. To recover a list 
        # in the form (b1s1, b2s1, ..., b1s2, ..., b1sN, ...), we can simply append, since we 
        # preserve the format that we predict on every batch and then increase the offset.
        result = torch.stack([p for sublist in outputs for p in sublist], axis=0)

        return result

    def __call__(self, d: dict):
        if self.data_splitter is not None:
            data_splitter = self.data_splitter
            assert data_splitter.offset is not None, 'cannot run validation with random data selection'

            sr = data_splitter.sr
            n_duration = data_splitter.duration * sr
            
            n_timepoints = d['x'].shape[-1]

            """
            To cover a recording of length 1 with duration a in (0, 1] and overlap
            b in [0, 1), we need to start a total of N = ceil( (1-a) / (a(1-b)) ) segments
            after starting the first one (so in total N + 1). 
            The reason for this is that each new segment covers a fraction 1-b of the 
            duration a before being overlapped with new segments, i.e. it covers a*(1-b). 
            We need to start new segments until a segment is sufficient to reach the end 
            of the recording, so we cover a range 1-a in total.
            """

            a = min([1, n_duration / n_timepoints])
            b = self.overlap

            N_segments = 1 + ceil((1-a) / (a * (1-b)))

            skip = ceil(n_timepoints // N_segments)# skip to do in each iteration

            # compute for each block of time
            d_copy = deepcopy(d)

            
            bs = d['x'].shape[0] # batch size
            logits = self.predict(bs, d_copy, skip, N_segments, n_duration).to(self.device)
            offsets = torch.tensor([[i*skip]*bs for i in range(N_segments)]).reshape((-1, )).to(self.device)
            lens = torch.concat([d['lens']]*N_segments, axis=0).to(self.device)
            logits = torch.where((lens < offsets).unsqueeze(axis=-1), -torch.inf, logits).to(self.device)
            # make a list of length N_segments where each element corresponds to 
            # predicting on the ith shifted version of x. 
            logits_buffer = [logits[i*bs:(i+1)*bs] for i in range(N_segments)]
            logits = self.compute_logits(logits_buffer)
            

            """
            # Old code kept for safety reasons
            logits_buffer = []
            for i in range(N_segments):
                offset = skip * i # skip i windows
                d['x'] = torch.roll(d_copy['x'], shifts=(offset), dims=-1)
                logits = self.forward_item(d)[0]
                # check against duration to avoid bogus predictions on padded data
                logits = torch.where(d['lens'].unsqueeze(axis=-1) < offset, -torch.inf, logits)
                logits_buffer.append(logits)
            
            logits = self.compute_logits(logits_buffer)
            """

            return logits.to(self.device), d['y'].float().to(self.device)

        else:
            return self.forward_item(d)
    
    def compute_logits(self, logits_buffer: list[torch.Tensor]):
        """
        Computes predictions from predictions from a window
        Parameters:
            logits_buffer:
                A list containing logits, which we would like to aggregate. 
        Returns:
            logits, a prediction based on all of the predictions from the segment
            The way to compute these is specified in self.policy
        """
        return self.policy(logits_buffer)
    
    def to_device(self, d: dict):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)
        return d

    def forward_item(self, d: dict):
        d = self.to_device(d)
        d = self.data_pipeline(d)
        x, y = d['x'], d['y'].float()
        logits = self.model(x)
        return logits, y
