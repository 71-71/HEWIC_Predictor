"""
This file defines the loss function.
Date: 2022-07-30
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.1,lam=5, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.lam = lam
    def forward(self, target, predict):
        pt = torch.sigmoid(predict) 
        loss = - (1-self.alpha) * (1 - pt) ** self.gamma * target * torch.log(pt) - (self.alpha)* pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


