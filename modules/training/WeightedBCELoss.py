

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.training.ComputeLossWeights import ComputeLossWeights
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WeightedBCELoss(nn.Module):
    def __init__(self, weight=None, beta=0.9, size_average=True):
        super(WeightedBCELoss, self).__init__()
        self.beta=beta
        self.weights = ComputeLossWeights(beta=self.beta).forward()
        self.WEIGHTSS = torch.Tensor(np.array(self.weights))

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        
        bs=int(len(targets)/len(self.weights)) #recovering the batch size
        weights1=(self.WEIGHTSS).repeat(bs)
        
        #first compute binary cross-entropy 
        BCE_weight = F.binary_cross_entropy(inputs.to(device), targets.to(device), reduction='mean',weight=weights1.to(device))
                       
        return BCE_weight.to(device)