import torch_audiomentations as tam
import torch.nn as nn 
    class Audiomentations(nn.Module):
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
        return self.augment(x,sample_rate=self.sample_rate)
