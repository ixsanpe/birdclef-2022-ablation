import torchaudio.transforms as T
import torch.nn as nn 
class SpecAugment(nn.Module):
    def __init__(self):
        #self.spec=spec
        super().__init__()
    def time_stretching(self,x,rate=1.2):
        stretch = T.TimeStretch()
        return stretch(x,rate)
        #return T.TimeStretch(self.spec,rate)
    def time_masking(self,x,time_mask_param=80):
        masking = T.TimeMasking(time_mask_param=time_mask_param)
        return masking(x)
        #return T.TimeMasking(self.spec,time_mask_param)
    def freq_masking(self,x,freq_mask_param=80):
        masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        return masking(x)
        #return T.FrequencyMasking(freq_mask_param)
    def combined(self,x,rate=1.2,time_mask_param=80,freq_mask_param=80):
        stretch = T.TimeStretch()
        time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        return time_masking(freq_masking(stretch(x,rate)))
    def forward(self,x):
        return self.time_masking(self.freq_masking(self.time_stretching(x)))
