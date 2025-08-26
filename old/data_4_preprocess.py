#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import pandas as pd


# In[21]:


base_path = "C:/Users/Ally/Desktop/sheng/data/4/data/PPG_FieldStudy"


# In[22]:


subject_dirs = [i for i in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, i))]


# In[28]:


chunks_all = []


# In[34]:


for j in range(len(subject_dirs)):
    BVP_path = os.path.join(base_path, subject_dirs[j] + "\\" + subject_dirs[j] + "_E4", "BVP.csv")
    EDA_path = os.path.join(base_path, subject_dirs[j] + "\\" + subject_dirs[j] + "_E4", "EDA.csv")
    TEMP_path = os.path.join(base_path, subject_dirs[j] + "\\" + subject_dirs[j] + "_E4", "TEMP.csv")

    # Read the CSV file
    df_BVP = pd.read_csv(BVP_path, header=None)
    df_EDA = pd.read_csv(EDA_path, header=None)
    df_TEMP = pd.read_csv(TEMP_path, header=None)

    # check whether the starting time are the same for 3 variables
    if df_BVP.iloc[0,0] == df_EDA.iloc[0,0] and df_BVP.iloc[0,0] == df_TEMP.iloc[0,0]:
        print(1)
    else:
        print(0)

    
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
    df['subject'] = subject_dirs[j]

    # chunk df into chunks of 4 seconds with 1 second overlap
    # Parameters
    fs = 32              # Sampling frequency (Hz)
    chunk_duration = 4   # seconds
    overlap_duration = 1 # seconds

    chunk_size = chunk_duration * fs
    overlap_size = overlap_duration * fs
    step_size = chunk_size - overlap_size  # How many rows to move forward for each new chunk

    # Store all chunks
    chunks = []

    # Go through the DataFrame
    for start in range(0, len(df) - chunk_size + 1, step_size):
        chunk = df.iloc[start : start + chunk_size].reset_index(drop=True)
        chunks.append(chunk)

    # Now `chunks` is a list of DataFrames, each 128 rows, overlapping by 32 rows

    chunks_all.append(chunks)
    


# In[35]:


chunks_all[0]


# In[36]:


for i, chunk in enumerate(chunks_all):
    pd.concat(chunk, axis = 0).reset_index(drop = True).to_csv(f'chunk_{i}.csv', index = False)


# In[ ]:




