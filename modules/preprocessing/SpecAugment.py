import torchaudio.transforms as T
import torch.nn as nn 
import torch
'''class SpecAugment(nn.Module):
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
    def forward(self, d: dict):
        #d['x']=self.time_masking(self.freq_masking(self.time_stretching(d['x'],rate=1.2),freq_mask_param=80),time_mask_param=80)
        d['x']=T.TimeMasking(T.FrequencyMasking(T.TimeStretch(d['x'],rate=1.2),freq_mask_param=80),time_mask_param=80)'''


# taken from https://pytorch.org/audio/stable/transforms.html
class SpecAugment(nn.Module):
    def __init__(
        self,
        stretch_factor=0.8
    ):
        super().__init__()
        self.spec_aug = torch.nn.Sequential(
                T.TimeStretch(stretch_factor, fixed_rate=True),
                T.FrequencyMasking(freq_mask_param=80),
                T.TimeMasking(time_mask_param=80),
            )
    def forward(self, d: dict):
        d['x']= self.spec_aug(d['x'])
        return d
