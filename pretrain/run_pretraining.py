#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 20:08:15 2025

@author: sheng
"""
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torch.utils.data import DataLoader, TensorDataset
from models.full_model import AutoencoderModel
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from models.plot_func import moving_average
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from zoneinfo import ZoneInfo  
import argparse
from utils.split_data import split_k_fold
import random

# ==== config ===

train_path = 'preprocessed_data/'
test_path = 'preprocessed_data/'
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='exp')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--ckpt_dir', type=str, default='ckpts/pretrain/exp')
parser.add_argument('--log_dir', type=str, default='runs/pretrain/exp')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

os.makedirs(args.ckpt_dir, exist_ok=True)
# preprocessed_path = '/Users/sheng/Documents/emotion_model_project/preprocessed_data/' # mac path
preprocessed_path = '/home/sheng/project/affective-computing/preprocessed_data/' # server path
# os.chdir(preprocessed_path)

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
k = 5 # num of folds in spliting



   

# === Load model ===
print("Initializing model...")

# set random seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    
    # set random seed again
    set_seed(args.seed + f)
    
    val_paths = paths_split[f]
    train_paths = [x for i, sub in enumerate(paths_split) if i != f for x in sub]

    # create checkpoint dir for each fold
    fold_run_name = f"{args.log_dir}_fold{f}_lr{args.lr}_bs{batch_size}"
    fold_log_dir = f"{fold_run_name}"
    writer = SummaryWriter(log_dir=fold_log_dir)
    
    '''
    set names of output log files, using time
    tz = ZoneInfo("Asia/Shanghai")  # 北京时间 = Asia/Shanghai = UTC+8
    stamp = datetime.now(tz).strftime("%Y%m%d-%H%M%S%z")  # 带+0800偏移
    run_name = f"{stamp}_lr{lr}_bs{batch_size}"
    writer = SummaryWriter(log_dir=f"runs/pretrain/{run_name}")
    '''
    
    fold_ckpt_dir = os.path.join(args.ckpt_dir, f"fold{f}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    
    # Initialize model and optimizer
    
    print("Initializing model...")
    model = AutoencoderModel(tcn_channels = [16, 32], 
                             transformer_dim = 96, 
                             transformer_heads = 3, 
                             transformer_layers = 1,
                             lstm_hidden = 64, 
                             lstm_layers = 2, 
                             output_dim = 3, 
                             seq_len = 960, 
                             dropout = 0.1)
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


        for i in range(len(train_paths)):
            
            print(f"train data file {i+1}")
            train_data = torch.load(os.path.join(preprocessed_path, train_paths[i]))[:,:,1:4]
            train_loader = DataLoader(TensorDataset(train_data, train_data), batch_size = batch_size, shuffle = True)
    
            # count total number of batches in this epoch
            len_train_loader = len(train_loader)
            total_num_train_loader += len_train_loader       
          
            
            for batch_idx, (inputs, _) in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx+1}/{len_train_loader}")
                
                inputs = inputs.to(device)
                outputs = model(inputs)
                assert_infinite(outputs, "train outputs") # debug INF
                loss = loss_fn(outputs, inputs)
                assert_infinite(loss, "train_loss") # debug INF
                optimizer.zero_grad()
                                
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

                loss.backward()
                bad = find_bad_grads(model)
                if bad:
                    print("[BAD GRAD PARAMS]:", bad[:10], " ... total:", len(bad))
                    # 可选：逐个打印范数看看大小
                    for n, p in model.named_parameters():
                        if p.grad is None: 
                            continue
                        if not torch.isfinite(p.grad).all():
                            print(n, "grad stats:", p.grad.min().item(), p.grad.max().item())
                    # 跳过该 batch，避免权重被污染
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # ==== find the bad gradients ===
                
                
                
                
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # debug INF
                if not torch.isfinite(torch.tensor(total_norm)):                    # debug INF
                    raise RuntimeError(f"Grad_norm NaN/Inf: {total_norm}")        # debug INF
                
                
                optimizer.step()
                
                bs = inputs.size(0)
                train_samples += bs
                total_train_loss += loss.item()*bs
                
                global_step += 1
                
                if batch_idx % 10 == 0:
                    writer.add_scalar("loss/train_batch", loss.item(), global_step)
                
        avg_train_loss = total_train_loss / max(1, train_samples)
        writer.add_scalar("loss/train_epoch", avg_train_loss, epoch)
        print(f"[Fold {f+1}/{k}] Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f}")
            
        # === Eval ===
        model.eval()
        total_val_loss, val_samples = 0.0, 0
        total_num_val_loader = 0
        
        print("Starting evaluation")
    
        with torch.no_grad():
    
            for j in range(len(val_paths)):

                if j % 50 == 0:
                    print(f"evaluate dataset {j+1}")
                val_data = torch.load(os.path.join(preprocessed_path, val_paths[j]))[:,:,1:4]
                val_loader = DataLoader(TensorDataset(val_data, val_data), batch_size = batch_size, shuffle=False)
        
                len_val_loader = len(val_loader)
                total_num_val_loader += len_val_loader
                
                for batch_idx, (inputs, _) in enumerate(val_loader):
                    if batch_idx % 10 == 0:
                        print(f"[Evaluate] dataset {j+1}, batch_idx {batch_idx+1}/{len_val_loader}")
                    
                    inputs = inputs.to(device)
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
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best.pt'))
    
    torch.save(model.state_dict(), os.path.join(fold_ckpt_dir, "final.pt"))
    print(f"[Fold {f+1}/{k}] Best Val Loss: {best_val:.6f} (ckpt: {os.path.join(fold_ckpt_dir, 'best.pt')})")            
    writer.add_hparams(
        {"lr": args.lr, "batch_size": args.batch_size, "seed": args.seed, "fold": f},
        {"hparam/best_val": best_val}
    )
    writer.close()
    
    fold_best_vals.append(best_val)

fold_best_vals = np.array(fold_best_vals, dtype=np.float64)
print("=" * 80)
print(f"K-Fold summary ({k} folds):")
print("Best val losses per fold:", fold_best_vals.round(6).tolist())
print(f"Mean ± Std: {fold_best_vals.mean():.6f} ± {fold_best_vals.std(ddof=1):.6f}")





