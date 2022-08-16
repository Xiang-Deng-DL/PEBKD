#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:33:45 2020

@author: xiangdeng
"""
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
    
class KL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KL, self).__init__()
        self.T = T

    def forward(self, y_s, p_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / p_s.shape[0]
        return loss
    