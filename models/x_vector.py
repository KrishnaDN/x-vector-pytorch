#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:32:09 2020

@author: krishna

The model parameters are same as the paper "Spoken Language Recognition using X-vectors"

"""

import torch.nn as nn
from models.tdnn import TDNN
import torch


class X_vector(nn.Module):
    def __init__(self, input_dim = 40, num_classes=8):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=9, dilation=1)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=15, dilation=2)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=15, dilation=1)
        self.tdnn5 = TDNN(input_dim=512, output_dim=1500, context_size=15, dilation=3)
        #### Frame levelPooling
        self.segment6 = nn.Linear(3000, 512)
        self.bn_segment6 = nn.BatchNorm1d(num_features=512, affine=False)
        self.segment7 = nn.Linear(512, 512)
        self.bn_segment7 = nn.BatchNorm1d(num_features=512, affine=False)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.bn_segment6(self.segment6(stat_pooling))
        x_vec = self.bn_segment7(self.segment7(segment6_out))
        predictions = self.softmax(self.output(x_vec))
        return predictions,x_vec



        