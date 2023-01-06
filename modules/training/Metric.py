from typing import Callable, Optional
import torch 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

"""
During the project, we discoveded some inconsistencies between torchmetrics' documentation
and implementation (see our presentation from Nov 24th and https://torchmetrics.readthedocs.io/en/stable/classification/precision.html ). 
Weary of bugs, we decided to implement these key metrics ourselves
"""

class PrecisionMacro():    
    def __init__(self, thresh=.5):
        self.thresh = thresh

    @staticmethod
    def score(pred: torch.Tensor, y: torch.Tensor, thresh=.5):
        assert pred.shape == y.shape 
        # check where pred and y agree
        pred = torch.where(pred.double() > thresh, 1, 0)
        y = torch.where(y.double() > thresh, 1, 0)
        if pred.sum() < 1: return torch.ones(y.shape[-1]).to(DEVICE)
        agree = torch.sum(torch.where(torch.logical_and(pred==y, y > 0), 1, 0), axis=0)

        if torch.all(agree == 0): return agree
        # divide by the number of preds
        n_preds = pred.sum(axis=0)
        score = agree / n_preds
        # return
        return score
    
    def __call__(self, pred: torch.Tensor, y: torch.Tensor):
        score = PrecisionMacro.score(pred, y, self.thresh)
        return score.float().nanmean()

class RecallMacro():  
    def __init__(self, thresh=.5):
        self.thresh = thresh

    @staticmethod  
    def score(pred: torch.Tensor, y: torch.Tensor, thresh=.5):
        assert pred.shape == y.shape 
        # check where pred and y agree
        pred = torch.where(pred.double() > thresh, 1, 0)
        y = torch.where(y.double() > thresh, 1, 0)
        if y.sum() < 1: return torch.ones(y.shape[-1]).to(DEVICE)
        agree = torch.sum(torch.where(torch.logical_and(pred==y, y > 0), 1, 0), axis=0)

        if torch.all(agree == 0): return agree
        # divide by the number of ys
        n_preds = y.sum(axis=0)
        score = agree / n_preds
        # return
        return score 
    
    def __call__(self, pred: torch.Tensor, y: torch.Tensor):
        score = RecallMacro.score(pred, y, self.thresh)
        return score.float().nanmean()

class F1Macro():
    def __init__(self, thresh=.5):
        self.thresh = thresh
    
    def __call__(self, pred: torch.Tensor, y: torch.Tensor):
        prec = PrecisionMacro.score(pred, y, self.thresh)
        rec = RecallMacro.score(pred, y, self.thresh)

        prec = torch.where(torch.isnan(prec), 0., prec.double())
        rec = torch.where(torch.isnan(rec), 0., rec.double())

        f1 = 2*prec*rec/(prec + rec)
        # consider only the birds which were present in preds or in y
        ids = torch.argwhere(torch.logical_or(y > 0, pred > self.thresh))
        # preds to consider
        f1 = torch.stack(
            [
                f1[a] for a in ids[:, 1]
            ], 
            axis=-1
        )

        # F1 is nan if and only if prec==rec==0. In this case, we set it to 0. 
        f1 = torch.where(torch.isnan(f1), 0., f1)
        return f1.float().nanmean()
