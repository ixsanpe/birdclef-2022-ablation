from typing import Callable, Optional
import torch 

class Metric():
    def __init__(self, name: str, metric: Callable):
        """
        A class that can compute a metric and also maintains a name for that metric
        """
        super().__init__()
        self.name = name 
        self.metric = metric
    
    def __call__(self, x, y):
        return self.metric(x, y)

class PickyScore():
    def __init__(self, cls: Callable, tol: float=.5, device: Optional[str]=None):
        """
        This class attempts to overcome the short comings of the torchmetrics
        used on many classes, when many are not present. Using e.g. 
        torchmetrics.classification.MultilabelF1Score with a macro average,
        if y_pred = y = 0, this affects the score negatively.

        Therefore, using this PickyScore we evaluate the torchmetric scores
        only on those samples that either are present (y = 1) or are predicted
        as present (pred = 1).

        The class takes as an argument cls, which is the torchmetric to use.
        Parameters:
            cls:
                The class to instantiate in order to compute the metric. E.g.
                torchmetrics.classification.MultilabelF1Score
            tol:
                The threshold for saying that a prediction should be 1
            device:
                None or the device on which we train. If None, we use cuda, if
                cuda is available.
        """
        self.tol = tol
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.cls = cls
    
    def __call__(self, pred: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the metric on classes that have a positive label or prediction
        Parameters:
            pred:
                tensor of predictions (soft or hard)
            y:
                Labels
        Returns:
            The macro averaged score derived from self.cls
        """
        # relevant ids
        ids = torch.argwhere(torch.logical_or(y > 0, pred > self.tol))
        # ys to consider
        test_y = torch.stack(
            [
                y[:, a] for a in ids[:, 1] 
                # ids[:, 1] gives the relevant classes where someone predicted a 1
            ], 
            axis=-1
        )
        # preds to consider
        test_pred = torch.stack(
            [
                (pred[:, a] > self.tol).int() for a in ids[:, 1]
            ], 
            axis=-1
        )
        # create an instance of the relevant metric
        num_classes = test_pred.shape[-1]
        score = self.cls(
            num_labels=num_classes, 
            topk = 1, 
            average='macro' 
        ).to(self.device) 
        # Compute the score
        retval = score(test_pred, test_y)
        return retval 