#!/bin/bash

# set date
date="2021-02-14"

# unload any modules that carried over from your command line session
module purge all

# load modules you need to use
module load python/anaconda3.6

# activate virtual environment 
conda config --prepend envs_dirs /home/hwi3319/anaconda3/envs
source activate twitter_vaccine_env


# get the sentiment for vaccine text (wo distribution) using the trained model
mkdir -p ./data/sentiment/
cd ./code/
python3 predict_xlnet.py "part${date}"

