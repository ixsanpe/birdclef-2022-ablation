import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
#from modules.training.ComputeLossWeights import ComputeLossWeights
import numpy as np

ALPHA = 0.25
GAMMA = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        BCE = F.binary_cross_entropy(inputs.to(device), targets.to(device), reduction='mean') #TODO: add the weights with beta here
        #BCE_unweight = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE.to(device))
        focal_loss = alpha * (1-BCE_EXP.to(device))**gamma * BCE.to(device)
                       
        return focal_loss.to(device)'''

#As implemented here: https://amaarora.github.io/2020/06/29/FocalLoss.html#alpha-and-gamma
#Reduction = None !!!
class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=ALPHA, gamma=GAMMA):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
    #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE_loss = F.binary_cross_entropy(inputs.to(device), targets.to(device), reduction='none')
        at = (self.alpha.gather(0, targets.type(torch.int64))).to(device)
        pt = torch.exp(-BCE_loss.to(device))
        F_loss = at*(1-pt.to(device))**self.gamma * BCE_loss.to(device)
        return (torch.mean(F_loss)).to(device)