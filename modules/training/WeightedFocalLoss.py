import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.training.ComputeLossWeights import ComputeLossWeights
import numpy as np

weights = ComputeLossWeights(beta=0.9).forward()
WEIGHTS = torch.Tensor(np.array(weights))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ALPHA = 0.25
GAMMA = 2

#Primary unweighted version
'''class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE_WEIGHT = F.binary_cross_entropy(inputs, targets, reduction='mean',weight=WEIGHTS) #TODO: add the weights with beta here
        BCE_unweight = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE_unweight)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE_WEIGHT
                       
        return focal_loss'''

#USING NLL: we should add a final log-softmax layer to the model
'''class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
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
        BCE_WEIGHT = F.nll_loss(inputs, torch.Tensor.long(targets), reduction='mean',weight=weights1) #TODO: add the weights with beta here
        BCE_unweight = F.nll_loss(inputs, torch.Tensor.long(targets), reduction='mean')
        BCE_EXP = torch.exp(-BCE_unweight)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE_WEIGHT
                       
        return focal_loss'''

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha = ALPHA, gamma=GAMMA,weight=None, size_average=True):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

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
        BCE_WEIGHT = F.binary_cross_entropy(inputs.to(device), targets.to(device), reduction='none',weight=weights1.to(device)) #TODO: add the weights with beta here
        BCE_unweight = F.binary_cross_entropy(inputs.to(device), targets.to(device), reduction='none')
        at = (self.alpha.gather(0, targets.type(torch.int64))).to(device)
        BCE_EXP = torch.exp(-BCE_unweight.to(device))
        focal_loss = torch.mean(at * (1-BCE_EXP.to(device))**self.gamma * BCE_WEIGHT.to(device))
                        
        return focal_loss.to(device)
