import torch.nn as nn

class TransformApplier(nn.Module):
    """
    Maintains a list of transforms to apply to input data. When called, it applies all of these transforms.
    """

    def __init__(
        self, 
        transform_list: list
    ): 
        super().__init__()
        self.transforms = nn.Sequential(*transform_list)
    
    def forward(self, x):
        return self.transforms(x)
