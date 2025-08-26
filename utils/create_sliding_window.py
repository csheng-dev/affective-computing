#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:56:50 2025

@author: sheng
"""
import torch


def create_sliding_window(data, window_size, step_size):
    windows = []
    for i in range(0, data.shape[0] - window_size + 1, step_size):
        window = data[i:i+window_size]
        windows.append(window.unsqueeze(0))
    if not windows:
        return None
    return torch.cat(windows, dim=0)        
        
    