# Processing data of SemEval 2013-2016 Task 4 Subtask A for training
## Download the SemEval data from https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip?file_subpath=%2F2017_English_final%2FDOWNLOAD%2FSubtask_A
## and put in to ../data/SemEval_data/original
import sys
import os
import pandas as pd

# get all the data files
all_files = []
for file in os.listdir('../data/SemEval_data/original'):
    if file.endswith('A.tsv') or file.endswith('A.txt'):
        all_files.append(os.path.join('../data/SemEval_data/original',file))

# process before reading
def combine_col(data):
    # only runs once since it modifies file
    df = pd.read_csv(data, sep='\t', header = None)
    ids = df.iloc[:,0].astype(str)+df.iloc[:,1].astype(str)
    df.iloc[:,0] = ids
    df = df.drop(df.columns[1], axis = 1)
    df.to_csv(data, sep=  '\t', index = False, header = None)

combine_col(all_files[2])
combine_col(all_files[10])

# read in each of the data file
df_lst = []
for file in all_files:
    df = pd.read_csv(file, sep='\t', usecols = [0,1,2], header = None)
    print('shape of df: ', df.shape)
    df_lst.append(df)

all_df = pd.concat(df_lst, axis = 0)
all_df.columns = ['id', 'label', 'text']

# save combined data
all_df.to_csv('../data/SemEval_data/processed/all_SemEval_data.csv',
              index = False)