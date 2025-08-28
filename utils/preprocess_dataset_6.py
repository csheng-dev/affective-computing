#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import torch
from create_sliding_window import create_sliding_window


# In[3]:

# Parameters
fs = 32              # Sampling frequency (Hz)
chunk_duration = 30   # seconds
overlap_duration = 29 # seconds
# root_path = '/Users/sheng/Documents/emotion_model_project/data' # path on mac
root_path = '/home/sheng/project/affective-computing/data' # path on server


def preprocess_data_6(fs, chunk_duration, overlap_duration, root_path):

    # Set the path to the target directory
    base_path = root_path + "/6/Stress_dataset"

    # Get a list of all subdirectories in "1"
    subject_dirs = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    # Print each subdirectory name
    for subdir in subject_dirs:
        print(subdir)
    
    


    # In[5]:

    no_chunk = 0
    for j in range(len(subject_dirs)):
        print(subject_dirs[j])
        
        term_dirs_all = os.listdir(os.path.join(base_path, subject_dirs[j]))
        print(term_dirs_all)
        term_dirs = [item for item in term_dirs_all if not item.endswith("Store")]
        print(term_dirs)
        
        for k in range(len(term_dirs)):
            print(term_dirs[k])
            BVP_path = os.path.join(base_path, subject_dirs[j], term_dirs[k], "BVP.csv")
            EDA_path = os.path.join(base_path, subject_dirs[j], term_dirs[k], "EDA.csv")
            TEMP_path = os.path.join(base_path, subject_dirs[j], term_dirs[k], "TEMP.csv")

            # print(BVP_path)
            # print(EDA_path)
            # print(TEMP_path)
        
            # Read the CSV file
            df_BVP = pd.read_csv(BVP_path, header=None)
            df_EDA = pd.read_csv(EDA_path, header=None)
            df_TEMP = pd.read_csv(TEMP_path, header=None)

            '''
            if df_BVP.iloc[0,0] == df_EDA.iloc[0,0] and df_BVP.iloc[0,0] == df_TEMP.iloc[0,0]:
                print(1)
                else:
                    print(0)
                    '''

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

            # Step 3: combine columns
            df= pd.concat([df_BVP_cut, df_EDA_cut, df_TEMP_cut], axis=1)
            
            # assign column names
            df.columns = ['BVP', 'EDA', 'TEMP']

            # set index as a column named "time"
            df = df.reset_index().rename(columns={'index': 'time'})
            # df['subject'] = subject_dirs[j]
            print(df.dtypes)
            print(df.head())
            df = df.astype(float)  # or .astype(np.float32)
            df = torch.tensor(df.values, dtype=torch.float32)
            
            chunk_size = chunk_duration * fs
            overlap_size = overlap_duration * fs
            step_size = chunk_size - overlap_size  # How many rows to move forward for each new chunk
            
            # Store all chunks
            chunks = []

            # Go through the DataFrame
            chunks = create_sliding_window(df, chunk_size, step_size)
            print('no_chunk = ', no_chunk)
            if chunks is None:
                next
            else:
                # path on mac
                # torch.save(chunks, f'/Users/sheng/Documents/emotion_model_project/preprocessed_data/dataset_6_chunk_{no_chunk}_{subject_dirs[j]}.pt')
                # path on server
                torch.save(chunks, f'//home/sheng/project/affective-computing/preprocessed_data/dataset_6_chunk_{no_chunk}_{subject_dirs[j]}.pt')

                no_chunk += 1
            
        '''
        for i, chunk in enumerate(chunks_all):
            try:
                pd.concat(chunk, axis = 0).reset_index(drop = True).to_csv(f'/Users/sheng/Documents/emotion_model_project/preprocessed_data/dataset_6_chunk_{no_chunk}.csv', index = False)
                print("no_chunk = ", no_chunk)
                no_chunk += 1
            except ValueError:
                print("no object to concatenate")
                pass
        '''    
preprocess_data_6(fs, chunk_duration, overlap_duration, root_path)




