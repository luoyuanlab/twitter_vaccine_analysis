import os
import pandas as pd
import sys

sentiment_path = '../data/sentiment'
cleaned_path = '../data/cleaned'
deepface_path ='../data/deepface'
pregnancy_path = '../data/pregnant_vaccine_text_wo_distribution'
merged_path = '../data/merged'

part = sys.argv[1]

# sentiment
sentiment_user_id = []
sentiment_tweet_id = []
sentiment = []
with open(os.path.join(sentiment_path, 'sentiment_part%s.csv' % (part)), 'r') as sentiment_file:
    for lines in sentiment_file:
        try:
            lst = lines.split('|$|')
            sentiment_user_id.append(lst[1])
            sentiment_tweet_id.append(lst[2])
            sentiment.append(lst[3])
        except:
            pass
sentiment_df = pd.DataFrame({'user_id': sentiment_user_id,
                             'tweet_id': sentiment_tweet_id,
                             'sentiment': sentiment})
# time and place
meta_user_id = []
meta_tweet_id = []
created_at = []
user_location = []
coordinates = []
place = []
with open(os.path.join(cleaned_path, 'tweets_en_part%s.csv' % (part)), 'r') as meta_file:
    for lines in meta_file:
        try:
            lst = lines.split('|$|')
            if len(lst) == 14:
                meta_user_id.append(lst[0].replace('\n', ''))
                meta_tweet_id.append(lst[1])
                created_at.append(lst[9])
                user_location.append(lst[10])
                coordinates.append(lst[11])
                place.append(lst[12].replace('\n', ''))
        except:
            pass
meta_df = pd.DataFrame({'user_id': meta_user_id,
                        'tweet_id': meta_tweet_id,
                        'created_at': created_at,
                        'user_location': user_location,
                        'coordinates': coordinates,
                        'place': place})

merged1 = pd.merge(sentiment_df, meta_df,
                   how='left',
                   on=['tweet_id', 'user_id'])

del sentiment_df
del meta_df


# pregnancy
read_text = []
clean_text = []
read_user_id_pregnancy = []
read_tweet_id_pregnancy = []
with open(os.path.join(pregnancy_path, 'pregnant_vaccine_text_wo_distribution_part%s.csv'%(part)), 'r') as pregnancy_file:
    for lines in pregnancy_file:
        lst = lines.split('|$|')
        read_text.append(lst[0])
        clean_text.append(lst[1])
        read_user_id_pregnancy.append(lst[2])
        read_tweet_id_pregnancy.append(lst[3].replace('\n',''))
    pregnancy_df = pd.DataFrame({'user_id':read_user_id_pregnancy,
                                 'tweet_id': read_tweet_id_pregnancy,
                                 'pregnancy_text': read_text

                                 })
merged2 = pd.merge(merged1, pregnancy_df,
                   how='left',
                   on=['user_id', 'tweet_id'])

del merged1
del pregnancy_df
merged2['pregnancy_related'] = merged2['pregnancy_text'].isna()==0

# deepface
deepface_user_id = []
deepface_age = []
deepface_gender = []
deepface_race = []

with open(os.path.join(deepface_path, 'deepface_part%s.csv' % (part)), 'r') as deepface_file:
    for lines in deepface_file:
        lst = lines.split(',')
        deepface_user_id.append(lst[0])
        deepface_age.append(lst[1])
        deepface_gender.append(lst[6])
        deepface_race.append(lst[13])

deepface_df = pd.DataFrame({'user_id':deepface_user_id,
                             'deepface_age': deepface_age,
                             'deepface_gender': deepface_gender,
                             'deepface_race': deepface_race
                             })

deepface_df = deepface_df.drop_duplicates()
merged = pd.merge(merged2, deepface_df,
                   how='left',
                   on=['user_id'])
del deepface_df

merged.to_csv(os.path.join(merged_path, 'merged_part%s.csv' % (part)))
