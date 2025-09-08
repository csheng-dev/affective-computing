#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:11:15 2025

@author: sheng
"""

import os
import pandas as pd
import os
import pandas as pd
import torch
from create_sliding_window import create_sliding_window
import math



# root_path = '/Users/sheng/Documents/emotion_model_project/data' # path on Mac
root_path = '/home/sheng/project/affective-computing/data' # path on server


fs = 32              # Sampling frequency (Hz)
chunk_duration = 30   # seconds
overlap_duration = 29 # seconds

def preprocess_data_3(fs, chunk_duration, overlap_duration, root_path):
    
    base_path = root_path + '/3/Data/'
    
    # Get a list of all subdirectories in "1"
    subject_dirs = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    subject_dirs
    
    # Print each subdirectory name
    for subdir in subject_dirs:
        print(subdir)
    
    for i in range(len(subject_dirs)):
        subdir = subject_dirs[i]
        subject_subdirs = [s for s in os.listdir(os.path.join(base_path, subject_dirs[i])) if not s.endswith('.DS_Store')]
        print(subject_subdirs)
        
        for j in range(len(subject_subdirs)):
            subject_subdir_path = os.path.join(base_path, subject_dirs[i], subject_subdirs[j])
            print(subject_subdir_path)
    
            BVP_path = os.path.join(subject_subdir_path, "BVP.csv")
            EDA_path = os.path.join(subject_subdir_path, "EDA.csv")
            TEMP_path = os.path.join(subject_subdir_path, "TEMP.csv")

            # Read the CSV file
            df_BVP = pd.read_csv(BVP_path, header=None)
            df_EDA = pd.read_csv(EDA_path, header=None)
            df_TEMP = pd.read_csv(TEMP_path, header=None)
            
            '''
            if df_BVP.iloc[0,0] == df_EDA.iloc[0,0] and df_BVP.iloc[0,0] == df_TEMP.iloc[0,0]:
                print("Time matches")
            else:
                print("Time doesn't match")
            '''
            
            
            # ignore the 1st and 2nd row
            df_BVP = df_BVP.iloc[2:].reset_index(drop=True)
            df_EDA = df_EDA.iloc[2:].reset_index(drop=True)
            df_TEMP = df_TEMP.iloc[2:].reset_index(drop=True)

            # downsample BVP to 32 hz
            df_BVP = df_BVP.iloc[::2].reset_index(drop=True)
            df_EDA = df_EDA.loc[df_EDA.index.repeat(8)].reset_index(drop=True)
            df_TEMP = df_TEMP.loc[df_TEMP.index.repeat(8)].reset_index(drop=True)
            
            # combine all three df into one data frame
            # Step 1: Find the minimum number of rows
            min_rows = min(len(df_BVP), len(df_EDA), len(df_TEMP))
        
            # Step 2: Cut each dataframe to the minimum number of rows
            df_BVP_cut = df_BVP.iloc[:min_rows].reset_index(drop=True)
            df_EDA_cut = df_EDA.iloc[:min_rows].reset_index(drop=True)
            df_TEMP_cut = df_TEMP.iloc[:min_rows].reset_index(drop=True)
            
            print(j)
            print(df_EDA.head())
            # Step 3: combine columns
            df= pd.concat([df_BVP_cut, df_EDA_cut, df_TEMP_cut], axis=1)
            
            # assign column names
            df.columns = ['BVP', 'EDA', 'TEMP']
            
            # set index as a column named "time"
            df = df.reset_index().rename(columns={'index': 'time'})
            
            print(df.head())
            df = df.astype(float)  # or .astype(np.float32)
            
            df = torch.tensor(df.values, dtype=torch.float32)

            # chunk df into chunks of given seconds with given seconds' overlap
            
            chunk_size = chunk_duration * fs
            overlap_size = overlap_duration * fs
            step_size = chunk_size - overlap_size  # How many rows to move forward for each new chunk
            
            # Store all chunks
            
            
            chunks = create_sliding_window(df, chunk_size, step_size)

            # path on mac            
            # torch.save(chunks, f'/Users/sheng/Documents/emotion_model_project/preprocessed_data/dataset_3_chunk_{subject_subdirs[j]}_{subject_dirs[i]}.pt') 
            # path on server
            torch.save(chunks, f'/home/sheng/project/affective-computing/preprocessed_data/dataset_3_chunk_{subject_subdirs[j]}_{subject_dirs[i]}.pt')
            

    
preprocess_data_3(fs, chunk_duration, overlap_duration, root_path)   
    
    