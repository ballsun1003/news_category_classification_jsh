import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re

df = pd.read_csv("./data/news_titles.csv")
df.info()
df = df.drop_duplicates(subset=['title', 'category'])

bad_titles = df.groupby('title')['category'].nunique()
bad_titles = bad_titles[bad_titles > 1].index
df = df[~df['title'].isin(bad_titles)]
print(df.head(30))
print(df.category.value_counts())

X = df.title
Y = df.category

# okt = Okt()
# okt_x = okt.morphs(X[0])
# print(okt_x)
#
# komoran = Komoran()
# komoran_x = komoran.morphs(X[0])
# print(komoran_x)

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:5])
label = encoder.classes_
print(label)
with open("./data/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
onehot_y = to_categorical(labeled_y)
print(onehot_y[:5])

# cleaned_x = re.sub('[^가-힣]', ' ', X[0])
# print(cleaned_x)
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
    '및', '듯', '뿐'
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

    if i % 1000 == 0:
        print(i)

print(X[:30])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
tokened_x = tokenizer.texts_to_sequences(X)
print(tokened_x)
wordsize = len(tokenizer.word_counts)+1
print(wordsize)
max = 0
for sentence in tokened_x:
    if max < len(sentence):
        max = len(sentence)
print(max)
with open('./data/tokenizer_max{}.pkl'.format(max), 'wb') as f:
    pickle.dump(tokenizer, f)

x_pad = pad_sequences(tokened_x, maxlen=max)
print(x_pad[:5])

x_train, x_test, y_train, y_test = train_test_split(x_pad, onehot_y, test_size=0.1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
np.save('./data/x_train.npy', x_train)
np.save('./data/y_train.npy', y_train)
np.save('./data/x_test.npy', x_test)
np.save('./data/y_test.npy', y_test)

#14999
#27