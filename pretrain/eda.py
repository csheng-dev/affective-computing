#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 19:53:59 2025

@author: sheng
"""

import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset, Subset
from models.full_model import AutoencoderModel
import matplotlib.pyplot as plt
from models.plot_func import moving_average

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from zoneinfo import ZoneInfo  
import argparse
from utils.split_data import split_k_fold
import random

from utils.experiment_utils import init_experiment

# ===== config =====

train_path = 'preprocessed_data/'
test_path = 'preprocessed_data/'
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='exp')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

# preprocessed_path = '/Users/sheng/Documents/emotion_model_project/preprocessed_data/' # mac path
preprocessed_path = '/home/sheng/project/affective-computing/preprocessed_data/' # server path

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
k = 5 # num of folds in spliting

# === Functions and classes needed ===
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# === prepare ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(4210)

paths_split = split_k_fold(k) # get list of k elements, each element contains paths for the fold

f = 1
set_seed(4210 + f)
val_paths = paths_split[f]
train_paths = [x for i, sub in enumerate(paths_split) if i != f for x in sub]

# concate all data file into one big torch tensor
train_ls = []
for p in train_paths:
    ds_path = os.path.join(preprocessed_path, p)
    ds = torch.load(ds_path)[:,:,1:4]
    train_ls.append(ds)
train_ds = torch.cat(train_ls, dim=0)

val_ls = []
for p in val_paths:
    ds_path = os.path.join(preprocessed_path, p)
    ds = torch.load(ds_path)[:,:,1:4]
    val_ls.append(ds)
val_ds = torch.cat(val_ls, dim=0)

def check_batch_stats(x, tag):
    x = x.detach().float()
    B, T, C = x.shape
    flat = x.view(-1, C)
    mean = flat.mean(0)
    std  = flat.std(0, unbiased=False)
    p99  = torch.quantile(flat, 0.99, dim=0)
    print(f"[{tag}] mean={mean.tolist()}  std={std.tolist()}  p99={p99.tolist()}  max_abs={flat.abs().max().item():.3f}")

# 在 seed=4210, fold=2 的训练/验证循环里，加：
if batch_idx in (0, 1, 2):   # 前几个 batch 看看就行
    check_batch_stats(inputs, f"fold2/train/b{batch_idx}")







