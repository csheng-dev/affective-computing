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




# ==== config ===
train_path = 'preprocessed_data/'
test_path = 'preprocessed_data/'
epochs = 5
batch_size = 64
lr = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set names of output log files
tz = ZoneInfo("Asia/Shanghai")  # 北京时间 = Asia/Shanghai = UTC+8
stamp = datetime.now(tz).strftime("%Y%m%d-%H%M%S%z")  # 带+0800偏移
run_name = f"{stamp}_lr{lr}_bs{batch_size}"
writer = SummaryWriter(log_dir=f"runs/pretrain/{run_name}")

# === Devide preprocessed data into train/test set === 

id_train_dataset = [6]
id_test_dataset = [1]
train_paths = []
test_paths = []
# preprocessed_path = '/Users/sheng/Documents/emotion_model_project/preprocessed_data/' # mac path
preprocessed_path = '/home/sheng/project/affective-computing/preprocessed_data/' # server path
names = os.listdir(preprocessed_path)
names = [name for name in names if not name.endswith('DS_Store')] # delete the redundant file ending with "DS_Store"
names_split = [name.split('_') for name in names]
for i in range(len(names_split)):
    if int(names_split[i][1]) in id_train_dataset:
        train_paths.append(preprocessed_path + names[i])
    if int(names_split[i][1]) in id_test_dataset:
        test_paths.append(preprocessed_path + names[i])

    

# === Load model ===
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
train_batch_losses = []
train_epoch_losses = []
test_epoch_losses = []
global_step = 0  # counts training batches


for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}")
    
    total_num_train_loader = 0
    total_train_loss = 0.0  # CHANGED
    
    # for i in range(len(train_paths)):
        
    # For DEBUG --
    for i in range(3):
    # For DEBUG --
    
        print(f"train data file {i+1}")
        train_data = torch.load(train_paths[i])[:,:,1:4]
        train_loader = DataLoader(TensorDataset(train_data, train_data), batch_size = batch_size, shuffle = True)

        # for DEBUG --
        small_dataset = train_data[:64*3]
        train_loader = DataLoader(TensorDataset(small_dataset, small_dataset), batch_size = batch_size, shuffle = True)
        # for DEBUG --

        # count total number of batches in this epoch
        len_train_loader = len(train_loader)
        total_num_train_loader += len_train_loader       
    
        print("Training")
        model.train()        
        
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx %% 100 == 0:
                print(f"Batch {batch_idx+1}/{len_train_loader}")
            
            # print(f"[input shape] {inputs.shape}")
            inputs = inputs.to(device)
            outputs = model(inputs)
            # print(f"[input shape] {inputs.shape}")
            # print(f"[output shape] {outputs.shape}")
            loss = loss_fn(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # === logging per-batch ===
            train_batch_losses.append(loss.item()) # ADDED
            total_train_loss += loss.item()
            global_step += 1  # ADDED
            
            writer.add_scalar("loss/train_batch", loss.item(), global_step)
            
    avg_train_loss = total_train_loss / max(1, total_num_train_loader)
    train_epoch_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}")
    writer.add_scalar("loss/train_epoch", avg_train_loss, epoch)
    # === Evaluation ===
    model.eval()
    total_test_loss = 0.0
    total_num_test_loader = 0
    print("Starting evaluation")
    with torch.no_grad():
    
        for j in range(len(test_paths)):
        # for DEBUG
        # for j in range(2):
        # for DEBUG
            print(f"evaluate dataset {j+1}")
            test_data = torch.load(test_paths[j])[:,:,1:4]
            test_loader = DataLoader(TensorDataset(test_data, test_data), batch_size = batch_size, shuffle=False)
 
            # for DEBUG --
            # small_dataset = test_data[:64*2]
            # test_loader = DataLoader(TensorDataset(small_dataset, small_dataset), batch_size = batch_size, shuffle = False)
            # for DEBUG --
    
            len_test_loader = len(test_loader)
            total_num_test_loader += len_test_loader
            
            for batch_idx, (inputs, _) in enumerate(test_loader):
                if batch_idx %% 100 == 0:
                    print(f"dataset {j+1}, batch_idx {batch_idx+1}/{len_test_loader}")
                
                inputs = inputs.to(device)
                outputs = model(inputs)
                # print(f"[input shape] {inputs.shape}")
                # print(f"[output shape] {outputs.shape}")
                loss = loss_fn(outputs, inputs)
                total_test_loss += loss.item()
                
    avg_test_loss = total_test_loss / max(1, total_num_test_loader)
    test_epoch_losses.append(avg_test_loss)
    print(f"Epoch [{epoch+1}/{epochs}] | Test Loss: {avg_test_loss}")
    writer.add_scalar("loss/test_epoch", avg_train_loss, epoch)


