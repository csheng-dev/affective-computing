#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:11:31 2025

@author: sheng
"""

import torch  # Import PyTorch main package
import torch.nn as nn  # Import PyTorch neural network module

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()  # Initialize the parent nn.Module
        # First 1D convolution layer
        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
        #                       stride=stride, padding=padding, dilation=dilation)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=4, dilation=2)
        self.relu1 = nn.ReLU()  # ReLU activation after first conv
        self.dropout1 = nn.Dropout(dropout)  # Dropout after first activation
        # Second 1D convolution layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()  # ReLU activation after second conv
        self.dropout2 = nn.Dropout(dropout)  # Dropout after second activation
        # Downsample input if in_channels != out_channels for residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()  # Initialize weights

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)  # He initialization for conv1
        nn.init.kaiming_normal_(self.conv2.weight)  # He initialization for conv2
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)  # He initialization for downsample if exists

    def forward(self, x):
        out = self.conv1(x)  # Pass input through first conv
        out = self.relu1(out)  # Apply ReLU
        out = self.dropout1(out)  # Apply dropout
        out = self.conv2(out)  # Pass through second conv
        out = self.relu2(out)  # Apply ReLU
        out = self.dropout2(out)  # Apply dropout
        res = x if self.downsample is None else self.downsample(x)  # Residual connection
        print("out shape: ", out.shape)
        print("res shape: ", res.shape)
        return out + res  # Add residual to output

class TCNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()  # Initialize parent nn.Module
        layers = []  # List to hold temporal blocks
        print("num_channels: ", num_channels)
        num_levels = len(num_channels)  # Number of temporal blocks
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # Input channels for this block
            out_channels = num_channels[i]  # Output channels for this block
            # Add a temporal block to the list
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                    padding=(kernel_size-1)*dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)  # Stack all temporal blocks sequentially

    def forward(self, x):
        return self.network(x)  # Pass input through the TCN network
