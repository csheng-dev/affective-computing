#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 20:08:58 2025

@author: sheng
"""

def pretrain_autoencoder(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = torch.nn.MSELoss()
    
    
    
    
    
    pass