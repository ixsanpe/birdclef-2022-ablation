import audiomentations as am
import torch.nn as nn 

#old one    
'''class Audiomentations(nn.Module):
    def __init__(self,
                transformations,
                sample_rate=32000
                ):
        self.transformations=transformations
        self.sample_rate=sample_rate
        super().__init__()
        self.augment=am.Compose(self.transformations)  

    def forward(self, x):
        if isinstance(x, tuple):
            return self.forward_x(x[0]), x[1]
        else:
            return self.forward_x(x)

    def forward_x(self,x):
        return self.augment(x,sample_rate=self.sample_rate)'''

#right one
'''class Audiomentations(nn.Module):
    def __init__(self,
                transformations,
                sample_rate=32000
                ):
        self.transformations=transformations
        self.sample_rate=sample_rate
        super().__init__()
        self.augment=am.Compose(self.transformations)  

    def forward(self, d: dict):
        t=d['x'].reshape([d['x'].shape[0],1,d['x'].shape[1]])
        c = self.augment(t, sample_rate=self.sample_rate)
        #d['x'] = self.augment(d['x'], sample_rate=self.sample_rate)
        d['x']=c.reshape([d['x'].shape[0],d['x'].shape[1]])
        return d '''

#try without dictionaries for augment_data
class Audiomentations(nn.Module):
    def __init__(self,
                transformations,
                sample_rate=32000
                ):
        self.transformations=transformations
        self.sample_rate=sample_rate
        super().__init__()
        self.augment=am.Compose(self.transformations)  

    def forward(self, data):
        #t=d['x'].reshape([d['x'].shape[0],1,d['x'].shape[1]])
        data = self.augment(data, sample_rate=self.sample_rate)
        #d['x'] = self.augment(d['x'], sample_rate=self.sample_rate)
        #d['x']=c.reshape([d['x'].shape[0],d['x'].shape[1]])
        return data 