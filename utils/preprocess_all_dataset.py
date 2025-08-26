#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:12:07 2025

@author: sheng
"""

import pandas as pd
import os


from preprocess_dataset_1 import preprocess_data_1
from preprocess_dataset_6 import preprocess_data_6


# Parameters
fs = 32              # Sampling frequency (Hz)
chunk_duration = 30   # seconds
overlap_duration = 29 # seconds
root_path = '/Users/sheng/Documents/emotion_model_project/data'



# preprocess_data_1(fs, chunk_duration, overlap_duration, root_path)
# preprocess_data_6(fs, chunk_duration, overlap_duration, root_path)

# split datasets for training and testing during pretraining process

train_id = [6]
test_id = [1]













