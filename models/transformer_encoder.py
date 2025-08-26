#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:11:53 2025

@author: sheng
"""

import torch 
import torch.nn as nn

# position encoding function

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        i = torch.arange(0, d_model, 2).unsqueeze(0) # (1, d_model/2)
        div = torch.exp(-torch.log(torch.tensor(10000)) * i / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe) # declare pe is not a parameter
        
    def forward(self, x_lbd):  # x_lbd: (L, B, d_model)
        L = x_lbd.size(0)
        return x_lbd + self.pe[:L].unsqueeze(1)  # (L, 1, d_model) broadcast over B


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)  # Linear layer to project input to model dimension
        # add positional enconding
        self.pos_enc = SinusoidalPE(model_dim)
        
        # Create a single transformer encoder layer
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=dropout
        )
        # Stack multiple encoder layers to form the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_linear(x)  # Project input to model dimension
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, model_dim)
        x = self.pos_enc(x)
        out = self.transformer_encoder(x)  # Pass through transformer encoder
        out = out.transpose(0, 1)  # Convert back to (batch, seq_len, model_dim)
        return out  # Return encoded output

