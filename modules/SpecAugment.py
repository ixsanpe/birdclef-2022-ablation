import torchaudio.transforms as T
class SpecAugment():
    def __init__(self,spec):
        self.spec=spec
    def time_stretching(self,rate=1.2):
        stretch = T.TimeStretch()
        return stretch(self.spec,rate)
        #return T.TimeStretch(self.spec,rate)
    def time_masking(self,time_mask_param=80):
        masking = T.TimeMasking(time_mask_param=time_mask_param)
        return masking(self.spec)
        #return T.TimeMasking(self.spec,time_mask_param)
    def freq_masking(self,freq_mask_param=80):
        masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        return masking(self.spec)
        #return T.FrequencyMasking(freq_mask_param)
    def combined(self,rate=1.2,time_mask_param=80,freq_mask_param=80):
        stretch = T.TimeStretch()
        time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        return time_masking(freq_masking(stretch(self.spec,rate)))

#Use it for one datapoint at a time, like this:
'''for spec in data:
    apply = SpecAugment(spec=spec)
    stretched=apply.time_masking()
    freq_masked=apply.freq_masking()
    time_masked=apply.time_masking()
    combined=apply.combined()'''
