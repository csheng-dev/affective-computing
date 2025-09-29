#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 19:52:29 2025

@author: sheng
"""



from torch.utils.data import ConcatDataset, Dataset

class LazyTensorDataset(Dataset):
    """ 每个文件一个Dataset，按需加载，避免一次性占用内存 """
    def __init__(self, file_path, use_channels=(1,2,3)):
        self.file_path = file_path
        self.use_channels = use_channels
        # 读取一次形状信息（也可以在 __getitem__ 时再懒加载）
        t = torch.load(file_path, map_location="cpu")  # [N, T, C_all]
        self.N = t.shape[0]
        del t

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        t = torch.load(self.file_path, map_location="cpu")  # 简单起见每次load；更高级可缓存
        x = t[idx][:, self.use_channels]  # [T, 3]
        return x, x

# 组装 train/val ConcatDataset
def make_concat_dataset(paths):
    dsets = []
    for p in paths:
        full = os.path.join(preprocessed_path, p)
        dsets.append(LazyTensorDataset(full, use_channels=(1,2,3)))
    return ConcatDataset(dsets)

train_ds = make_concat_dataset(train_paths)
val_ds   = make_concat_dataset(val_paths)

# 单个 DataLoader（可复现）
base_seed = args.seed + f*1000 + epoch
train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True,  base_seed=base_seed, num_workers=0)
val_loader   = make_loader(val_ds,   batch_size=batch_size, shuffle=False, base_seed=base_seed, num_workers=0)
