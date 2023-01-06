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
        """
        Apply self.transformations to d['x']
        """
        t = d['x'].reshape([d['x'].shape[0], 1, d['x'].shape[1]]) # proper shape for tam
        c = self.augment(t, sample_rate=self.sample_rate)
        d['x'] = c.reshape([d['x'].shape[0], d['x'].shape[1]])
        return d 

