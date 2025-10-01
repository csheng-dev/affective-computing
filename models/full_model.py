#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:12:17 2025

@author: sheng
"""

import torch  
import torch.nn as nn  
from models.tcn_encoder import TCNEncoder  # Import TCNEncoder
from models.transformer_encoder import TransformerEncoder  # Import TransformerEncoder
from models.lstm_decoder import LSTMDecoder  # Import LSTMDecoder

class AutoencoderModel(nn.Module):
    def __init__(
            self, 
            tcn_channels, 
            transformer_dim, 
            transformer_heads, 
            transformer_layers, 
            lstm_hidden, lstm_layers, 
            output_dim, 
            seq_len=960, 
            dropout=0.1
    ):
        super().__init__() 
        # Three separate TCN encoders, one for each variable
        self.tcn1 = TCNEncoder(num_inputs=1, num_channels=tcn_channels, kernel_size=3, stride=1, dropout=dropout)
        self.tcn2 = TCNEncoder(num_inputs=1, num_channels=tcn_channels, kernel_size=3, stride=1, dropout=dropout)
        self.tcn3 = TCNEncoder(num_inputs=1, num_channels=tcn_channels, kernel_size=3, stride=1, dropout=dropout)
        self.seq_len = seq_len
        self.tcn_out_channels = tcn_channels[-1]  # Output channels from each TCN
        # Transformer encoder
        self.transformer = TransformerEncoder(input_dim=3*self.tcn_out_channels, model_dim=transformer_dim, num_heads=transformer_heads, num_layers=transformer_layers, dropout=dropout)
        # LSTM decoder
        self.decoder = LSTMDecoder(input_dim=transformer_dim, hidden_dim=lstm_hidden, num_layers=lstm_layers, output_dim=output_dim, dropout=dropout)

    def forward(self, x):
        # x: (batch, seq_len, 3) - input with 3 variables
        # Split input into three variables
        x1 = x[..., 0].unsqueeze(-1)  # (batch, seq_len, 1)
        x2 = x[..., 1].unsqueeze(-1)  # (batch, seq_len, 1)
        x3 = x[..., 2].unsqueeze(-1)  # (batch, seq_len, 1)
        # Transpose for Conv1d: (batch, 1, seq_len)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1)
        # Pass each variable through its own TCN encoder
        out1 = self.tcn1(x1)  # (batch, tcn_out_channels, seq_len)
        out2 = self.tcn2(x2)  # (batch, tcn_out_channels, seq_len)
        out3 = self.tcn3(x3)  # (batch, tcn_out_channels, seq_len)
        # Concatenate along channel dimension: (batch, 3*tcn_out_channels, seq_len)
        cat = torch.cat([out1, out2, out3], dim=1)
        # Transpose for transformer: (batch, seq_len, 3*tcn_out_channels)
        cat = cat.permute(0, 2, 1)
        # Pass through transformer encoder
        encoded = self.transformer(cat)  # (batch, seq_len, transformer_dim)
        # Pass through LSTM decoder
        decoded, _ = self.decoder(encoded)  # (batch, seq_len, output_dim)
        return decoded  # Return reconstructed sequence


