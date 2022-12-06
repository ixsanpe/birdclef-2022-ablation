import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.training.ComputeLossWeights import ComputeLossWeights
import numpy as np

weights = ComputeLossWeights(beta=0.9).forward()
WEIGHTS = torch.Tensor(np.array(weights))


class WeightedBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(WeightedBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bs=int(len(targets)/len(weights)) #recovering the batch size
        weights1=WEIGHTS.repeat(bs)
        
        #first compute binary cross-entropy 
        #print('targets:',targets,targets.size())
        #print(targets)
        BCE_weight = F.binary_cross_entropy(inputs, targets, reduction='mean',weight=weights1) #TODO: add the weights with beta here
                       
        return BCE_weight
