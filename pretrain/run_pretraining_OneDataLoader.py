#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 19:52:29 2025

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
# parser.add_argument('--ckpt_dir', type=str, default='ckpts/pretrain/exp')
parser.add_argument('--log_dir', type=str, default='runs/pretrain/exp')
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()


# ===== create config =====
                         
config = {
    "name": args.run_name,
    "seed": args.seed,
    "model": {
        "tcn_channels": [16, 32],
        "transformer_dim": 96,
        "transformer_heads": 3,
        "transformer_layers": 1,
        "lstm_hidden": 64,
        "lstm_layers": 2,
        "output_dim": 3,
        "seq_len": 960,
        "dropout": 0.1
    },    
    "train": {
        "batch_size": args.batch_size,
        "epoch": args.epochs,
        "lr": args.lr
    }
} 

exp_dir = init_experiment(config)
print(f"[INFO] Experiment directory has been created: {exp_dir}")


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

    torch.use_deterministic_algorithms(True, warn_only=False)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

class LazyTensorDataset(Dataset):
    def __init__(self, file_path, use_channels=(1,2,3)):
        self.file_path = file_path
        self.use_channels = use_channels
        t = torch.load(file_path, map_location="cpu")  
        # t is shape: [N, T, c_all]
        self.N = t.shape[0]
        del t
        self._cache = None
    def _ensure_loaded(self):
        if self._cache is None:
            self._cache = torch.load(self.file_path, map_location="cpu")
            
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        self._ensure_loaded()
        x = self._cache[idx][:, self.use_channels]   # [T, 3]
        return x, x


# concate train/val ConcatDataset
def make_concat_dataset(paths, root_dir):
    dsets = []
    for p in paths:
        full = os.path.join(root_dir, p)
        dsets.append(LazyTensorDataset(full, use_channels=(1,2,3)))
    return(ConcatDataset(dsets))

    
def make_loader(dataset, batch_size, shuffle, base_seed, num_workers=4, 
                prefetch_factor=4, pin_memory=True):
    '''
    build a reproducible DataLoader:
    - control the shuffle randomness
    - fix the randomness of worker's state
    '''    
    
    # create a generator of the DataLoader level, to control randomness of shuffle
    g = torch.Generator()
    g.manual_seed(base_seed) # make sure the shuffle order is the same
    
    # work_init_fn: make sure in the process of multiple processes (num_worker>0)
    # that the seed of each numpy/random is the same for each worker
    def worker_init_fn(worker_id):
        seed = base_seed + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    # construct DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,      # control the shuffle of DataLoader
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=prefetch_factor                
    )
    
    return loader
    
# === find the bad gradients
def find_bad_grads(model):
    bad_list = []
    for n, p in model.named_parameters():
        if p.grad is None: 
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            bad_list.append(n)
    return bad_list



# === Load model ===
print("Initializing model...")



    
# === prepare ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

paths_split = split_k_fold(k) # get list of k elements, each element contains paths for the fold

fold_best_vals = []

def assert_infinite(t, name):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} has NaN/INF")
 


# start k fold cross validation
for f in range(k):
    print("=" * 80)
    print(f"[fold {f+1}/{k}]")
    
    fold_dir = os.path.join(exp_dir, f"fold_{f}")
    os.makedirs(fold_dir, exist_ok=True)
    ckpt_dir = os.path.join(fold_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # set random seed again
    set_seed(args.seed + f)

    val_paths = paths_split[f]
    train_paths = [x for i, sub in enumerate(paths_split) if i != f for x in sub]
    
    train_ds = make_concat_dataset(train_paths, preprocessed_path)
    val_ds = make_concat_dataset(val_paths, preprocessed_path)

    

    # create checkpoint dir for each fold
    tb_dir = os.path.join(exp_dir, "tensorboard", f"fold_{f}_lr{args.lr}_bs{batch_size}")
    writer = SummaryWriter(log_dir=tb_dir)
    # fold_ckpt_dir = os.path.join(args.ckpt_dir, f"fold{f}")
    # os.makedirs(fold_ckpt_dir, exist_ok=True)
    
    print("Initializing model...")
    model = AutoencoderModel(tcn_channels = config['model']['tcn_channels'], 
                             transformer_dim = config['model']['transformer_dim'], 
                             transformer_heads = config['model']['transformer_heads'], 
                             transformer_layers = config['model']['transformer_layers'],
                             lstm_hidden = config['model']['lstm_hidden'], 
                             lstm_layers = config['model']['lstm_layers'], 
                             output_dim = config['model']['output_dim'], 
                             seq_len = config['model']['seq_len'], 
                             dropout = config['model']['dropout'])
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    # train several epochs, record early/best of the fold
    best_val = float("inf")
    global_step = 0
    
    for epoch in range(epochs):
        print(f"[Fold {f+1}/{k}] Starting epoch {epoch+1}/{epochs}")
    
    
        # === Train ===
        model.train()
        
        total_num_train_loader = 0
        total_train_loss, train_samples = 0.0, 0
    
        base_seed = args.seed + f*1000 + epoch
        
        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, base_seed=base_seed, num_workers=4)
        val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, base_seed=base_seed, num_workers=4)
        
        # count total number of batches in this epoch
        len_train_loader = len(train_loader)
        
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"Fold {f+1}/{k} Batch {batch_idx+1}/{len_train_loader}") # keep track of progress
                
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            assert_infinite(outputs, "train outputs") # debug INF
            loss = loss_fn(outputs, inputs)
            assert_infinite(loss, "train_loss") # debug INF
            optimizer.zero_grad() 
            
            loss.backward()
    
            bad = find_bad_grads(model)
            
            '''
            # detect bad gradients and stop training
            if bad:
                print("[BAD GRAD PARAMS]:", bad[:10], " ... total:", len(bad))

                for n, p in model.named_parameters():
                    if p.grad is None: 
                        continue
                    if not torch.isfinite(p.grad).all():
                        print(n, "grad stats:", p.grad.min().item(), p.grad.max().item())
                raise RuntimeError("Catch error") 
            
            '''
            
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # norm clipping gradients
            if not torch.isfinite(torch.tensor(total_norm)):                    # debug INF
                raise RuntimeError(f"Grad_norm NaN/Inf: {total_norm}")    # debug INF
                
            
            optimizer.step()
            
            bs = inputs.size(0)
            train_samples += bs
            total_train_loss += loss.item()*bs
            
            # plot loss vs batch_idx 
            if batch_idx % 10 == 0:
                writer.add_scalar("loss/train_batch", loss.item(), global_step)
            
            global_step += 1

        avg_train_loss = total_train_loss / max(1, train_samples)
        writer.add_scalar("loss/train_epoch", avg_train_loss, epoch)
        print(f"[Fold {f+1}/{k}] Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f}")

    
        # === Eval ===
        model.eval()
        total_val_loss, val_samples = 0.0, 0
        len_val_loader = len(val_loader)
        
        print("Starting evaluation")
        
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(val_loader):
                if batch_idx % 100 == 0:
                    print(f"[Fold {f+1}/{k}] Epoch {epoch+1} [Evaluate] batch_idx {batch_idx+1}/{len_val_loader}")
                
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                
                assert_infinite(outputs, "val outputs")    # debug INF
     
                loss = loss_fn(outputs, inputs)
                
                assert_infinite(loss, "val loss")   # debug INF
                
                bs = inputs.size(0)
                val_samples += bs
                total_val_loss += loss.item() * bs
                
            avg_val_loss = total_val_loss / max(1, val_samples)
            writer.add_scalar("loss/val_epoch", avg_val_loss, epoch)
            print(f"[Fold {f+1}/{k}] Epoch {epoch+1} | Val Loss: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
            
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))
    
    print(f"[Fold {f+1}/{k}] Best Val Loss: {best_val:.6f} (ckpt: {os.path.join(ckpt_dir, 'best.pt')})")            
    
    writer.add_hparams(
        {"lr": args.lr, "batch_size": args.batch_size, "seed": args.seed, "fold": f},
        {"hparam/best_val": best_val}
    )
    
    
    fold_best_vals.append(best_val)
    
    print(f"Best val loss: {fold_best_vals}")

    writer.close()












        















