#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:36:49 2025

@author: sheng
"""

import numpy as np
import pandas as pd
import torch
import os
import random
from pathlib import Path

def split_k_fold(num_folds):
    k = num_folds # number of folders
    rng = random.Random(42) # fix the seed for following random shuffles
    
    # root_path = '/Users/sheng/Documents/emotion_model_project/preprocessed_data' # path on mac
    root_path = '/home/sheng/project/affective-computing/preprocessed_data' # path on server
    
    all_paths =  os.listdir(root_path)

    dataset_subject_split = [ x.split('_') for x in all_paths] 

    dataset_subject_split

    for i in range(len(all_paths)):
        if len(dataset_subject_split[i]) == 6:
            original_name = dataset_subject_split[i][-2] + '_' + dataset_subject_split[i][-1]
            
            dataset_subject_split[i] = dataset_subject_split[i][:-2] + [original_name]

    df = pd.DataFrame(dataset_subject_split, columns=['v1', 'ds_num', 'v3', 'v4', 'subject'])
    
    df.isna().sum()
    df = df.dropna()
        
    df['subject'] = df['subject'].apply(lambda x: x.replace('.pt', ''))

    subject_in_d_1 = df.loc[df['ds_num'] == '1','subject']
    s_1 = set(subject_in_d_1)

    for e in {'S11_a', 'S11_b', 'S16_a', 'S16_b', 'f14_a', 'f14_b'}:
        s_1.remove(e)
    s_1.add('f14')
    l_1 = list(s_1)
    rng.shuffle(l_1)
    folds = [l_1[i::k] for i in range(k)]
    folds
    for i in range(k):
        for j in ['S11', 'S16']:
            if j in folds[i]:
                folds[i].append(j+'_a')
                folds[i].append(j+'_b')
        if 'f14' in folds[i]:
            folds[i].append('f14_a')
            folds[i].append('f14_b')
            folds[i].remove('f14')
        
    folds_names = [[] for _ in range(k)] # list to store file names of k folds

    for (i, path) in enumerate(all_paths):
        # create k_folds with file names for dataset 1
        p = Path(path)
        p_stem = p.stem
        path_split = p_stem.split('_')

        for j in range(k):
            for (t, w) in enumerate(folds[j]):                
                if len(path_split) < 5:
                    next
                elif (path_split[1] == '1') and (folds[j][t] in path):
                    folds_names[j].append(path)


    # create k_folds with file names for dataset 2-5

    for ds_no in range(2,6):
        subject_in_d = df.loc[df['ds_num'] == str(ds_no),'subject']
        s = set(subject_in_d)
        lst = list(s)
        rng.shuffle(lst)
        folds = [lst[i::k] for i in range(k)]
        
        for (i, path) in enumerate(all_paths):
            p = Path(path)
            p_stem = p.stem
            path_split = p_stem.split('_')
            
            # create k_folds with file names for dataset i
            
            for j in range(k):
                if len(path_split) < 5:
                    continue
                if path_split[1] == str(ds_no) and path_split[4] in folds[j]:
                    folds_names[j].append(path)
        
    return(folds_names)


if __name__ == '__main__':
    split_k_fold(5)















