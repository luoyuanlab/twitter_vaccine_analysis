from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch

import sys
import re
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F

proportion = sys.argv[1]

BATCH_SIZE = 4
MAX_LEN = 256
EPOCHS = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['negative', 'neutral', 'positive']

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 3)
model = model.to(device)
model.load_state_dict(torch.load('../data/sentiment_model/xlnet_base_model_traintest8_maxlen%s_epoch%s_vaccine.bin'%(MAX_LEN,EPOCHS)))


def predict_sentiment(text):

    encoded_review = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = pad_sequences(encoded_review['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post",
                              padding="post")
    input_ids = input_ids.astype(dtype='int64')
    input_ids = torch.tensor(input_ids)

    attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor,
                                   truncating="post", padding="post")
    attention_mask = attention_mask.astype(dtype='int64')
    attention_mask = torch.tensor(attention_mask)

    input_ids = input_ids.reshape(1, MAX_LEN).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    outputs = outputs[0][0].cpu().detach()

    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim=-1)

    # print("Positive score:", probs[1])
    # print("Negative score:", probs[0])
    # print(f'Review text: {text}')
    # print(f'Sentiment  : {class_names[prediction]}')

    return class_names[prediction]


def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()  # .lower() not lower case here since xlnet is only available in cased


def del_http_user_tokenize(tweet):
  # delete [ \t\n\r\f\v]
  space_pattern = r'\s+'
  url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
               r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  mention_regex = r'@[\w\-]+'
  tweet = re.sub(space_pattern, ' ', tweet)
  tweet = re.sub(url_regex, '', tweet)
  tweet = re.sub(mention_regex, '', tweet)
  return tweet


f = open('../data/vaccine_text_wo_distribution/vaccine_text_wo_distribution_%s.csv'%(proportion),'r')
out = open('../data/sentiment/sentiment_%s.csv' % (proportion), 'w+')
s = '|$|'
for lines in f.readlines():
    try:
        lst = lines.split('|$|')
        read_text = clean_str(del_http_user_tokenize(lst[0]))
        read_user_id = lst[2].replace('\n', '')
        read_tweet_id = lst[3].replace('\n', '')
        sentiment = predict_sentiment(read_text)
        out_str = s.join([read_text, read_user_id, read_tweet_id, sentiment, '\n'])

        out.write(out_str)
    except:
        pass
out.close()
f.close()