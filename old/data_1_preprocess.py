#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import pandas as pd


# In[56]:


# Parameters
fs = 32              # Sampling frequency (Hz)
chunk_duration = 4   # seconds
overlap_duration = 1 # seconds
root_path = '/Users/sheng/Documents/emotion_model_project/data'



def preprocess_data_1(fs, chunk_duration, overlap_duration, root_path):

    # Set the path to the target directory

    base_path = root_path + '/1'

    # Get a list of all subdirectories in "1"
    subdirs = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    # Print each subdirectory name
    for subdir in subdirs:
        print(subdir)
    
    chunks_all = []


# In[ ]:





# In[44]:


    for i in range(3):
        dir_name = subdirs[i]
        dir_path = os.path.join(base_path, dir_name)
        subject_dirs = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        subject_dirs
        for j in range(len(subject_dirs)):
            subject_dirs[j]
            os.listdir(os.path.join(dir_path, subject_dirs[j]))
            BVP_path = os.path.join(dir_path, subject_dirs[j], "BVP.csv")
            EDA_path = os.path.join(dir_path, subject_dirs[j], "EDA.csv")
            TEMP_path = os.path.join(dir_path, subject_dirs[j], "TEMP.csv")

            # Read the CSV file
            df_BVP = pd.read_csv(BVP_path, header=None)
            df_EDA = pd.read_csv(EDA_path, header=None)
            df_TEMP = pd.read_csv(TEMP_path, header=None)

            # check whether the starting time are the same for 3 variables
            if df_BVP.iloc[0,0] == df_EDA.iloc[0,0] and df_BVP.iloc[0,0] == df_TEMP.iloc[0,0]:
                next
                else:
                    print(0)

            # found all starting time are the same


    # In[45]:
        
        
        # from above, found the starting time are the same for different variables for different subject
        # combine 3 variables and chunk them into 4 seconds with 1 second overlap
        for i in range(3):
            dir_name = subdirs[i]
            dir_path = os.path.join(base_path, dir_name)
            subject_dirs = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
            subject_dirs
            for j in range(len(subject_dirs)):
                subject_dirs[j]
                os.listdir(os.path.join(dir_path, subject_dirs[j]))
                BVP_path = os.path.join(dir_path, subject_dirs[j], "BVP.csv")
                EDA_path = os.path.join(dir_path, subject_dirs[j], "EDA.csv")
                TEMP_path = os.path.join(dir_path, subject_dirs[j], "TEMP.csv")
                
                # Read the CSV file
                df_BVP = pd.read_csv(BVP_path, header=None)
                df_EDA = pd.read_csv(EDA_path, header=None)
                df_TEMP = pd.read_csv(TEMP_path, header=None)
                
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


    # In[46]:


    len(chunks_all)


    # In[52]:
        
        
        for i, chunk in enumerate(chunks_all):
            pd.concat(chunk, axis = 0).reset_index(drop = True).to_csv(f'chunk_{i}.csv', index = False)


# In[53]:


os.getcwd()


# In[ ]:




