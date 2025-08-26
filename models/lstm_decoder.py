#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:12:04 2025

@author: sheng
"""

import torch  
import torch.nn as nn  

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super().__init__()  
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Linear layer to project LSTM output to desired output dimension
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        out, hidden = self.lstm(x, hidden)  # Pass input through LSTM
        out = self.output_linear(out)  # Project LSTM output to output dimension
        return out, hidden  # Return output and hidden state

