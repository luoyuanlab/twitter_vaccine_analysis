import json
import re
from tqdm import trange, tqdm
import os
import sys
sys.path.append('./')
from utilities import del_http_user_tokenize, clean_str

part = sys.argv[1]
if not os.path.exists("../data/full_text"):
    os.makedirs("../data/full_text")

with open("/projects/b1131/Hanyin/Twitter/extracted/%s_clean_extracted"%(part)) as tweet_file:
    for line in tqdm(tweet_file):
        dic = json.loads(line)

        try:
            read_text = dic['full_text']
            read_text = re.sub('\n', ' ', read_text) # remove newline
            read_text_clean = clean_str(del_http_user_tokenize(read_text))  # clean string

            read_user_id = dic['user']['id']
            read_tweet_id = dic['id']
            lang = dic['lang']

            if lang == 'en':
                s = '|$|'

                output_str = s.join([read_text, read_text_clean, str(read_user_id), str(read_tweet_id)])
                f = open('../data/full_text/full_text_en_part%s.csv'%(part), 'a')
                f.write(output_str)
                f.write('\n')
                f.close()
        except:
            pass


print('&' * 80)
print('finished loading information')
