import json
from tqdm import tqdm
import sys
import urllib.request
import os
from nameparser import HumanName

print(os.getcwd())
part = sys.argv[1]

if not os.path.exists("../data/profile_img/part_%s" % (part)):
    os.makedirs("../data/profile_img/part_%s" % (part))
if not os.path.exists("../data/cleaned"):
    os.makedirs("../data/cleaned")

with open("../data/extracted/%s_clean_extracted"%(part)) as tweet_file:
    for line in tqdm(tweet_file):
        dic = json.loads(line)

        try:
            name = dic['user']['name']

            read_user_id = dic['user']['id_str']
            read_tweet_id = dic['id_str']
            lang = dic['lang']
            created_at = dic['created_at']

        except:
            continue

        try:
            user_location = dic['user']['location']
        except:
            user_location = ''
        try:
            coordinates = dic['coordinates']
        except:
            coordinates = ''
        try:
            place = dic['place']
        except:
            place= ''
        try:
            profile_img_url = dic['user']['profile_image_url']
        except:
            pass

        if lang == 'en':
            s = '|$|'
            name_parsed = HumanName(name)  # parse name
            output_str = s.join([str(read_user_id), str(read_tweet_id),
                                 name_parsed.first, name_parsed.last, name_parsed.middle, name_parsed.title,
                                 name_parsed.suffix, name_parsed.nickname,
                                 lang, created_at, str(user_location),
                                 str(coordinates), str(place), str(profile_img_url)])
            f = open('../data/cleaned/tweets_en_part%s.csv'%(part), 'a')
            f.write(output_str)
            f.write('\n')
            f.close()
            # download profile image
            try:
                urllib.request.urlretrieve(profile_img_url,
                                           "../data/profile_img/part_%s/profile_img_%s.jpg" % (
                                               part, read_user_id))
            except:
                pass


print('&' * 80)
print('finished loading information')
