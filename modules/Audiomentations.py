import audiomentations as am
import torch.nn as nn 
'''class Audiomentations(nn.Module):
    def __init__(self,
        sample_rate=32000,
        min_amplitude=0.001,
        max_amplitude=0.015,
        p_Gauss=0.5,
        min_rate=0.8,
        max_rate=1.25,
        p_time=0.5,
        min_semitones=-4,
        max_semitones=4,
        p_Pitch=0.5,
        min_fraction=-0.5,
        max_fraction=0.5,
        p_Shift=0.5):
        super().__init__()
        self.augment=am.Compose([
        am.AddGaussianNoise(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p_Gauss),
        am.TimeStretch(min_rate=min_rate, max_rate=max_rate, p=p_time),
        am.PitchShift(min_semitones=min_semitones, max_semitones=max_semitones, p=p_Pitch),
        am.Shift(min_fraction=min_fraction, max_fraction=max_fraction, p=p_Shift),
        ])  
    def forward(self,x):
        return self.augment(x,sample_rate=self.sample_rate)''''
    
class Audiomentations(nn.Module):
    def __init__(self,
                 sample_rate=32000
                 transformations)
        self.transformations=transformations
        super().__init__()
        self.augment=am.Compose(self.transformations)  
    def forward(self,x):
        return self.augment(x,sample_rate=self.sample_rate)
