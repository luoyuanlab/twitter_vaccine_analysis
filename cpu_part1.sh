#!/bin/bash

# set date
date="2021-02-14"

# download raw daily data
mkdir -p ./data/original/
mkdir -p ./data/extracted/

cd ./data/original/

wget https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/"${date}"/"${date}"_clean-dataset.tsv.gz?raw=true
mv ./"${date}"_clean-dataset.tsv.gz?raw=true ./"${date}"_clean-dataset.tsv.gz
gunzip "${date}"_clean-dataset.tsv.gz 

# unload any modules that carried over from your command line session
module purge all

# load modules you need to use
module load python/anaconda3.6

# activate virtual environment 
conda config --prepend envs_dirs /home/hwi3319/anaconda3/envs
source activate twitter_vaccine_env

# hydrate
cd ../../code/hydrate/
python get_metadata.py -i ../../data/original/"${date}_clean-dataset.tsv" -o ../../data/extracted/"${date}_clean_extracted" -k ./api_keys.json -c tweet_id -m e

# get profile image and name (and other inforamtion)
cd ../
python3 get_profile_img.py "${date}"  

# get full text
python3 get_full_text.py "${date}"

# select vaccine text
python3 select_vaccine_text.py "${date}"

# select the distribution-related text in the vaccine text
python3 select_distribution.py "${date}"

# select the pregnancy related text in wo distribution vaccine text
python3 select_pregnant_text.py "${date}"

# get demographic information using deepface
python3 run_deepface.py "${date}"
