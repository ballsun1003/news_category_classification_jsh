import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re

df = pd.read_csv("./data/naver_headline_news_20260605.csv")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df.title
Y = df.category.replace({
    'Social': 'Society',
    'IT': 'IT_Science'
})

with open("./data/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

label_y = encoder.transform(Y)
onehot_y = to_categorical(label_y, num_classes=6)

okt = Okt()

DROP_POS = {
    'Josa',
    'Eomi',
    'PreEomi',
    'Punctuation',
    'KoreanParticle',
    'Exclamation',
    'Determiner',
    'Suffix',
}

KEEP_POS = {
    'Noun',
    'Verb',
    'Adjective',
    'Adverb',
    'Alpha',
    'Number',
    'Foreign',
}

DROP_ONE = {
    '것', '수', '등', '때', '곳', '거', '점',
    '중', '전', '후', '내', '외', '위', '뒤',
    '더', '또', '못', '및', '첫', '새'
}

KEEP_ONE = set('美中日北韓尹與野檢軍法警靑英佛獨印러')

X = list(X)

for i in range(len(X)):
    text = str(X[i])

    text = re.sub('[^가-힣A-Za-z0-9一-龥]', ' ', text)
    text = re.sub(' +', ' ', text).strip()
    text = text.lower()

    words = []

    for word, pos in okt.pos(text, norm=True, stem=True):
        word = word.strip()
        if word == '':
            continue
        if pos in DROP_POS:
            continue
        if pos not in KEEP_POS:
            continue
        if len(word) == 1:
            if word in DROP_ONE:
                continue
            if word in KEEP_ONE:
                words.append(word)
                continue
            if word.isdigit():
                words.append(word)
                continue
            continue
        words.append(word)
    X[i] = ' '.join(words)

print(X[:30])

with open("./data/tokenizer_max27.pkl", "rb") as f:
    tokenizer = pickle.load(f)
tokened_x = tokenizer.texts_to_sequences(X)
print(tokened_x[:5])

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 27:
        tokened_x[i] = tokened_x[i][:27]
x_pad = pad_sequences(tokened_x, maxlen=27)
print(x_pad[:5])

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 27:
        tokened_x[i] = tokened_x[i][:27]
x_pad = pad_sequences(tokened_x, maxlen=27)
print(x_pad[:5])

import glob
import os

results = []

for path in glob.glob("./model_*.h5"):
    model = load_model(path,safe_mode=False)
    scores = model.evaluate(x=x_pad, y=onehot_y, verbose=0)

    loss = scores[0]
    acc = scores[1]

    results.append((os.path.basename(path), loss, acc))

results.sort(key=lambda x: x[2], reverse=True)

for rank, (name, loss, acc) in enumerate(results, start=1):
    print(f"{rank}위 | {name} | loss={loss:.4f} | acc={acc:.4f}")