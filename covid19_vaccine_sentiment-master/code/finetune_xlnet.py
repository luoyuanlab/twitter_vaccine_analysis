# finetune XLNet on SemEval / annotated data

from transformers import XLNetTokenizer, XLNetModel, AdamW, get_linear_schedule_with_warmup, XLNetForSequenceClassification
import torch

import re
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import defaultdict
from collections import Counter

from torch import nn, optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

train = sys.argv[1] == 'train'
eval = sys.argv[1] == 'eval'

data = sys.argv[2] # 'semeval' or 'annotated'

BATCH_SIZE = 4
MAX_LEN = 256
EPOCHS = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if data == 'annotated':
    path_to_data = '../data/annotation/annotation_label_4500.csv'
elif data == 'semeval':
    path_to_data = '../data/SemEval_data/processed/all_SemEval_data.csv'
df = pd.read_csv(path_to_data, index_col = 0)
df.columns = ['text', 'tweet_id','label']
# remove duplicated records (for SemEval data)
# df = df.drop_duplicates(subset = 'id')

# Shuffle data
df = df.sample(frac=1)

n_words = []
for i in range(df.shape[0]):
    n_words.append(len(df['text'].iloc[i].split()))
print('Max number of words before cleaning: %s'%(str(max(n_words))))


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
    return string.strip()#.lower()

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

df['text'] = df['text'].apply(del_http_user_tokenize)
df['text'] = df['text'].apply(clean_str)

# Function to convert labels to number
def sentiment2label(sentiment):
    if sentiment == "negative":
        return 0
    elif sentiment == 'neutral':
        return 1
    elif sentiment == 'positive':
        return 2

# re-level label for semeval
# df['label'] = df['label'].apply(sentiment2label)
Counter(df['label'])

# List of class names.
class_names = ['negative', 'neutral', 'positive']

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

class SemEvalDataset(Dataset):

    def __init__(self, text, label, tokenizer, max_len):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.label[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor, truncating="post",
                                  padding="post")
        input_ids = input_ids.astype(dtype='int64')
        input_ids = torch.tensor(input_ids)

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor,
                                       truncating="post", padding="post")
        attention_mask = attention_mask.astype(dtype='int64')
        attention_mask = torch.tensor(attention_mask)

        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask.flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# train-val-test split
df_train, df_test = train_test_split(df, test_size=0.2, stratify = df['label'], random_state=7777)
df_test, df_val = train_test_split(df_test, test_size=0.2, stratify = df_test['label'], random_state=7777)

df_train.shape, df_val.shape, df_test.shape

# data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = SemEvalDataset(
    text=df.text.to_numpy(),
    label=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4,
      drop_last = True
  )


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 3)
model = model.to(device)
# load the model trained on semeval
model.load_state_dict(torch.load('../data/sentiment_model/xlnet_base_model_traintest8_maxlen%s.bin'%(MAX_LEN)))

# setting hyper-parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

# sanity check on one batch
# data = next(iter(val_data_loader))
# data.keys()
#
# input_ids = data['input_ids']
# attention_mask = data['attention_mask']
# label = data['label']
# print(input_ids.reshape(4,512).shape) # batch size x seq length
# print(attention_mask.shape) # batch size x seq length

# Define training step function
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    acc = 0
    counter = 0

    for d in data_loader:
        input_ids = d["input_ids"].reshape(BATCH_SIZE, MAX_LEN).to(device)
        attention_mask = d["attention_mask"].to(device)
        label = d["label"].to(device)

        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=label)
        loss = outputs[0]
        logits = outputs[1]

        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        label = label.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(label, prediction)

        acc += accuracy
        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter = counter + 1

    return acc / counter, np.mean(losses)

# Define evaluation step function
def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].reshape(BATCH_SIZE, MAX_LEN).to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=label)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            label = label.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = metrics.accuracy_score(label, prediction)

            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, np.mean(losses)

#Fine tune
# %%time

if train:
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} Train accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            device,
            len(df_val)
        )

        print(f'Val loss {val_loss} Val accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), '../data/sentiment_model/xlnet_base_model_traintest8_maxlen%s_epoch%s_vaccine.bin'%(MAX_LEN,EPOCHS))
            best_accuracy = val_acc
elif eval:
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=3)
    model = model.to(device)
    model.load_state_dict(torch.load('../data/sentiment_model/xlnet_base_model_traintest8_maxlen%s_epoch%s_vaccine.bin'%(MAX_LEN,EPOCHS)))

    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        device,
        len(df_test)
    )

    print('Test Accuracy :', test_acc)
    print('Test Loss :', test_loss)


    def get_predictions(model, data_loader):
        model = model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                text = d["text"]
                input_ids = d["input_ids"].reshape(BATCH_SIZE, MAX_LEN).to(device)
                attention_mask = d["attention_mask"].to(device)
                label = d["label"].to(device)

                outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=label)

                loss = outputs[0]
                logits = outputs[1]

                _, preds = torch.max(outputs[1], dim=1)

                probs = F.softmax(outputs[1], dim=1)

                review_texts.extend(text)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(label)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values


    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )
    print(metrics.classification_report(y_test, y_pred, target_names=class_names))