import torch.nn as nn 
import torch 
from modules import SelectSplitData, TransformApplier
from typing import Callable
import warnings
from math import ceil
from copy import deepcopy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Validator():
    def __init__(
            self, 
            data_pipeline, 
            model, 
            device: str, 
            overlap: float=.5, 
            bs_max=32,
            policy: Callable=lambda l: torch.stack(l, axis=-1).max(axis=-1).values, 
            scheme='new'
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
            scheme:
                only for debugging purposes
        """
        super().__init__()
        self.data_pipeline = data_pipeline
        self.model = model
        assert overlap < 1 and overlap >= 0, f'overlap has to be in [0, 1) but got {overlap=}'
        self.overlap = overlap
        self.policy = policy
        self.device = device
        self.bs_max = bs_max
        self.scheme = scheme 

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
        """
        For each offset, predict the logits on N_segments. Return a batch no greater
        than self.bs_max.
        Parameters:
            d_copy: 
                copy of the dict; it will be modified
            skip:
                the amount to skip between each window
            N_segments:
                The number of segments to predict on (no more than self.bs_max // bs due to copying)
            n_duration:
                number of timepoints taken by SelectSplitData
            offset:
                the offset index at which to start
        Returns:
            preds for (batch:s1, batch:s2, ..., batch:sN)
        """
        # Copies of x, offset by i*skip + offset_term at the ith element
        copies = [
            torch.roll(d_copy['x'], shifts=(offset+i)*skip, dims=-1)
            for i in range(N_segments)
        ]
        
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
        """
        Predict helper function for batched prediction
        Parameters:
            bs:
                batch size of the input
            d_copy:
                a copy of the original dict (in case we want to manipulate it)
            skip:
                The amount by which to jump in each segment
            N_segments:
                The number of segments neccessary to cover the input file (or batch thereof)
            n_duration:
                number of timepoints taken by SelectSplitData
        Returns:
            A list whose ith element is the prediction (for the whole batch) on the i-th semgent

        """
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
                offset=i*(self.bs_max//bs) # in each step, we make this much progress and skip it next time
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

            if self.scheme == 'old':
                logits_buffer = self.simple_prediction(skip, N_segments, d_copy, d)# .to(self.device)
            
            else:
                logits_buffer = self.batched_prediction(skip, N_segments, d_copy, d, n_duration)
            logits = self.compute_logits(logits_buffer)

            return logits.to(self.device), d['y'].float().to(self.device)

        else:
            return self.forward_item(d)
    
    def batched_prediction(self, skip, N_segments, d_copy, d, n_duration):
        """
        Batched way to predict. Use self.predict to predict on batches
        and aggregate those predictions by setting any that were shifted too
        much to -inf. 
        Returns: 
            list whose i-th element is the prediction for the i-th segment
        """
        bs = d['x'].shape[0] # batch size
        logits = self.predict(bs, d_copy, skip, N_segments, n_duration).to(self.device)

        # logits has shape (batch:segment1, ..., batch:segmentK)
        offsets = torch.tensor([[i*skip]*bs for i in range(N_segments)]).reshape((-1, )).to(self.device).double()
        lens = torch.concat([d['lens']]*N_segments, axis=0).to(self.device).double()

        logits = torch.where((lens < offsets).unsqueeze(axis=-1), -torch.inf, logits.double()).to(self.device)
        # make a list of length N_segments where each element corresponds to 
        # predicting on the ith shifted version of x. 
        logits_buffer = [logits[i*bs:(i+1)*bs] for i in range(N_segments)]
        return logits_buffer
    
    def simple_prediction(self, skip, N_segments, d_copy, d):
        """
        Simplest way to predict.
        We compute the number of segments and iterate over 0, ..., N_segments - 1
        and shift x by offset each time. When we shift too much (more than the 
        length of x), we fill the logits with -inf
        """
        # Old code kept for safety reasons
        logits_buffer = []
        for i in range(N_segments):
            offset = skip * i # skip i windows
            d['x'] = torch.roll(d_copy['x'], shifts=(offset), dims=-1)
            logits = self.forward_item(d)[0].double()
            # check against duration to avoid bogus predictions on padded data
            logits = torch.where(d['lens'].unsqueeze(axis=-1).double() < offset, -torch.inf, logits.to(self.device)).to(self.device)
            logits_buffer.append(logits)
        return logits_buffer

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
        return logits.to(self.device), y.to(self.device)

def first_and_final(l):
    """
    Base predictions on first and final windows of prediction only
    """
    relevant = [l[0], l[-1]]
    return torch.stack(relevant, axis=-1).mean(axis=-1).to(DEVICE)

def max_all(l):
    """
    Take the max of each window as the prediction
    """
    return torch.stack(l, axis=-1).max(axis=-1).values.to(DEVICE)

def max_thresh(l, thresh=-1):
    """
    Take the max over each window if it exceeds a threshold of thresh
    """
    res = max_all(l).double()
    return torch.where(res > thresh, res, -torch.inf).to(DEVICE)
