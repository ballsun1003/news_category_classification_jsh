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
    if i % 100 == 0:
        print(i)
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
import re
import pandas as pd
from keras.models import load_model
from keras import backend as K


# =========================
# 설정
# =========================

# .keras로 변환한 모델 폴더
MODEL_GLOB = "./models_keras/model_*.keras"

# 훈련 때 저장한 결과 CSV
TRAINING_RESULT_CSV = "./training_results.csv"

# 최신 헤드라인 평가 결과 저장 파일
EVAL_RESULT_CSV = "./headline_eval_results.csv"

# 로드 실패한 모델 저장 파일
SKIPPED_CSV = "./headline_eval_skipped.csv"


# =========================
# 파일명 정규화 함수
# =========================
def get_model_key(path):
    """
    .h5 / .keras 확장자가 달라도 같은 모델로 매칭하기 위한 key 생성.
    예:
    model_0.712345_model1_bat64_emb512_classW.h5
    model_0.712345_model1_bat64_emb512_classW.keras

    둘 다:
    model_0.712345_model1_bat64_emb512_classW
    """
    name = os.path.basename(str(path))
    name = name.replace(".h5", "")
    name = name.replace(".keras", "")
    return name


def parse_model_name(path):
    """
    파일명에서 모델번호, batch, embedding, class_weight 여부 추출.
    training_results.csv 매칭 실패해도 최소 정보는 CSV에 남기기 위함.
    """
    name = os.path.basename(str(path))

    model_no = None
    batch_size = None
    embed_dim = None
    class_weight = False

    m = re.search(r"_model(\d+)_bat(\d+)_emb(\d+)", name)
    if m:
        model_no = int(m.group(1))
        batch_size = int(m.group(2))
        embed_dim = int(m.group(3))

    if "classW" in name:
        class_weight = True

    return model_no, batch_size, embed_dim, class_weight


# =========================
# training_results.csv 로드
# =========================

train_df = None

if os.path.exists(TRAINING_RESULT_CSV):
    train_df = pd.read_csv(TRAINING_RESULT_CSV)

    # saved_path가 있으면 그걸 기준으로 매칭
    if "saved_path" in train_df.columns:
        train_df["model_key"] = train_df["saved_path"].apply(get_model_key)
    else:
        train_df["model_key"] = ""

    # 훈련 결과 컬럼 이름 충돌 방지용 prefix
    rename_map = {}
    for col in train_df.columns:
        if col not in ["model_key", "saved_path"]:
            rename_map[col] = "train_" + col

    train_df = train_df.rename(columns=rename_map)

    print("training_results loaded:", TRAINING_RESULT_CSV)
else:
    print("training_results.csv 없음. 평가 결과만 저장함.")


# =========================
# 모델 평가
# =========================

results = []
skipped = []

model_paths = sorted(glob.glob(MODEL_GLOB))

print("found models:", len(model_paths))

for path in model_paths:
    name = os.path.basename(path)
    model_key = get_model_key(path)

    model_no, batch_size, embed_dim, class_weight_from_name = parse_model_name(path)

    try:
        K.clear_session()

        model = load_model(path, compile=False)

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        scores = model.evaluate(
            x=x_pad,
            y=onehot_y,
            verbose=0
        )

        eval_loss = float(scores[0])
        eval_acc = float(scores[1])

        row = {
            "model_name": name,
            "model_path": path,
            "model_key": model_key,

            # 파일명에서 뽑은 정보
            "model_no_from_name": model_no,
            "batch_size_from_name": batch_size,
            "embed_dim_from_name": embed_dim,
            "class_weight_from_name": class_weight_from_name,

            # 최신 헤드라인 평가 결과
            "headline_loss": eval_loss,
            "headline_accuracy": eval_acc,
        }

        results.append(row)

        print(f"OK   | {name} | loss={eval_loss:.4f} | acc={eval_acc:.4f}")

    except Exception as e:
        err = f"{type(e).__name__}: {e}"

        skipped.append({
            "model_name": name,
            "model_path": path,
            "model_key": model_key,
            "error": err,
        })

        print(f"SKIP | {name} | {err}")


# =========================
# 평가 결과 DataFrame 생성
# =========================

eval_df = pd.DataFrame(results)

if len(eval_df) > 0:
    # 평가 정확도 기준 정렬
    eval_df = eval_df.sort_values("headline_accuracy", ascending=False)
    eval_df.insert(0, "headline_rank", range(1, len(eval_df) + 1))

    # training_results와 병합
    if train_df is not None and "model_key" in train_df.columns:
        eval_df = eval_df.merge(
            train_df,
            on="model_key",
            how="left"
        )

    # CSV 저장
    eval_df.to_csv(EVAL_RESULT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("HEADLINE EVAL RANKING")
    print("=" * 100)

    for _, row in eval_df.iterrows():
        print(
            f"{int(row['headline_rank'])}위 | "
            f"{row['model_name']} | "
            f"headline_loss={row['headline_loss']:.4f} | "
            f"headline_acc={row['headline_accuracy']:.4f}"
        )

    print("\n평가 결과 저장:", EVAL_RESULT_CSV)

else:
    print("평가 성공한 모델 없음.")


# =========================
# 실패 목록 저장
# =========================

skip_df = pd.DataFrame(skipped)

if len(skip_df) > 0:
    skip_df.to_csv(SKIPPED_CSV, index=False, encoding="utf-8-sig")
    print("스킵 목록 저장:", SKIPPED_CSV)