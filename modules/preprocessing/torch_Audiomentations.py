import torch_audiomentations as tam
import torch.nn as nn 
class torch_Audiomentations(nn.Module):
    def __init__(self,
                transformations,
                sample_rate=32000
                ):
        self.transformations=transformations
        self.sample_rate=sample_rate
        super().__init__()
        self.augment=tam.Compose(self.transformations)  

    def forward(self, d: dict):
        d['x'] = self.augment(d['x'], sample_rate=self.sample_rate)
        return d 

