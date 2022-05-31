#!/bin/bash

# set date
date="2021-02-14"

# unload any modules that carried over from your command line session
module purge all

# load modules you need to use
module load python/anaconda3.6

# activate virtual environment 
source /projects/p31384/virtualenv/twitter_vaccine_env/bin/activate

# merge all info
mkdir -p ./data/merged/
cd ./code/
python3 merge.py "${date}"  

# start elasticsearch
cd /projects/p31384/virtualenv/twitter_vaccine_env/elasticsearch-7.2.0
./bin/elasticsearch > elasticsearch.log 2>&1 &

# parse geo
cd /projects/p31384/twitter_vaccine/github/code
python3 parse_geo.py "${date}"  
