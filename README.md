# Sentiment towards COVID-19 vaccines on social media: a Twitter analysis study

This GitHub holds the codes and data for the paper *"Sentiment towards COVID-19 vaccines on social media: a Twitter analysis study"*.

## Folder structure

```
.
│   README.md
│   requirements.txt
│   example_run.sh    
│   cpu_part1.sh
│   gpu_part.sh
│   cpu_part2.sh
│
└───code
│   │   utilities.py
│   │   get_full_text.py
│   │   ...
│   │
│   └───hydrate
│       │   api_keys.json (modify this file to add your Twitter API credentials)
│       │   ...
│
└───data
    │   annotation
    │   cleaned
    │   deepface
    │   extracted
    │   full_text
    │   merged
    │   original
    │   pregenant_vaccine_text_wo_distribution
    │   profile_img
    │   SemEval_data
    │   sentiment
    │   sentiment_model 
    └─── xlnet_base_model_traintest8_maxlen256_epoch8_vaccine.bin (see downloaing detailed below)
    │   vaccine_text
    │   vaccine_text_wo_distribution
```

## Usage

1. Download trained weights for models
    * Download the finetuned weights for XLNet.
    ```shell
    cd twitter_vaccine_analysis
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16EzhgGW6j5IRj1jpiJt4mCLRXaKGqkyg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16EzhgGW6j5IRj1jpiJt4mCLRXaKGqkyg" -O ./data/sentiment_model/xlnet_base_model_traintest8_maxlen256_epoch8_vaccine.bin && rm -rf /tmp/cookies.txt
    ```
   
2. Add [Twitter API credentials](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api) to `./code/hyrate/api_keys.json` (see the [Folder structure](#folder-structure)).

3. Install required packages.
    ```shell
    pip install -r requirements.txt
    ```
4. Set the date of which the data you want in `example_run.sh`, `cpu_part1.sh`, `cpu_part2.sh`, and `gpu_part.sh`, then
    ```shell
    ./example_run.sh
    ```
   Notes:
   * The job was designed to run on HPC, change allocation and queue attributes before running. 
   * `gpu_part.sh` needs access to GPUs.


## Data

7,700 manually annotated tweets for sentiment classification can be found in the [data](data/annotation) folder.
