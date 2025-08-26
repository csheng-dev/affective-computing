#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 17:13:34 2025

@author: sheng
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm




class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
                
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                               dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                               dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        
        
    def forward(self, x):
        # print(f"[x shape] {x.shape}")
        out = self.conv1(x)
        # print(f"[after conv1] {out.shape}")
        out = self.chomp1(out)
        # print(f"[after chomp1] {out.shape}")
        out = self.relu1(out)
        # print(f"[after relu1] {out.shape}")
        out = self.dropout1(out)
        # print(f"[after dropout1] {out.shape}")
        
        out = self.conv2(out)
        # print(f"[after conv2] {out.shape}")
        out = self.chomp2(out)
        # print(f"[after chomp2] {out.shape}")
        out = self.relu2(out)
        # print(f"[after relu2] {out.shape}")
        out = self.dropout2(out)
        # print(f"[after dropout2] {out.shape}")
        
        res = x if self.downsample is None else self.downsample(x)
        # print(f"[res shape] {res.shape}")
        return self.relu(out + res)
        
class TCNEncoder(nn.Module):
    # num_inputs: an integer
    # num_channes: an array of integers
    def __init__(self, num_inputs, num_channels, kernel_size, stride, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i ==0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        
            self.network = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.network(x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        