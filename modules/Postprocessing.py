import torch
from torch import Tensor
import torch.nn as nn

class PredictionThreshold(nn.Module):

    def __init__(
        self,
        class_thresholds = 0.5,
        num_classes: int = 152
        ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if isinstance(class_thresholds, float):
            self.class_thresholds = Tensor(num_classes * [class_thresholds])
        elif isinstance(class_thresholds, list):
            self.class_thresholds = Tensor(class_thresholds)
        elif isinstance(class_thresholds, Tensor):
            self.class_thresholds = class_thresholds
        else:
            raise NotImplementedError('Not implemented for input type ' + str(type(class_thresholds)))
            
    def __call__(self, y_pred) -> Tensor:
        assert (y_pred.shape[-1] == self.class_thresholds.shape[-1]), f'y_pred does not have the correct shape: {y_pred.shape} != {self.class_thresholds.shape}'
        y_out = []
        if y_pred.dim() == 2:
            for y_slice in y_pred:
                y_out.append(torch.where(y_slice > self.class_thresholds, 1,0).int())
            y_out = torch.stack(y_out, dim=0)
        elif y_pred.dim() == 1:
            y_out = torch.where(y_pred > self.class_thresholds, 1,0).int()
        else:
            raise NotImplementedError('Class PredictionThreshold not implements for input tensor with %i dimensions' %(y_out.dim()))
        return y_out