import torch.nn as nn

class TransformApplier(nn.Module):
    """
    Maintains a list of transforms to apply to input data. When called, it applies all of these transforms.

    At the moment we could simply have used a nn.Sequential instead but we are maintaining this class
    in case we want to implement further functionalities not provided by nn.Sequential. 
    """

    def __init__(
        self, 
        transform_list: list
    ): 
        super().__init__()
        self.transforms = nn.Sequential(*transform_list)
    
    def forward(self, x):
        return self.transforms(x)
    
    def __getitem__(self, idx):
        return list(self.transforms)[idx]
