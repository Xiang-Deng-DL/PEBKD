#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:32:48 2020

@author: xiangdeng
"""

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class Softmax_T(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(Softmax_T, self).__init__()
        self.T = T

    def forward(self, y):    
        p = F.softmax(y/self.T, dim=1)
        return p