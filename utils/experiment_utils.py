#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 20:42:36 2025

@author: sheng
"""

import os, json, yaml, datetime, shutil

def init_experiment(config_dict, base_dir="experiment"):
    
    '''
    Initialize a directory to store configuration and parameters
    '''
    
    # 1. Create experiment ID, with time
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_id = f"{ts}_{config_dict.get('name', 'exp')}"
    exp_dir = os.path.join(base_dir, exp_id)
    
    # 2. Create folder structure
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "folds"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)
    
    # 3. Store the configuration
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
        
    
    return exp_dir


