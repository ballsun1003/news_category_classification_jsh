import os
import re
import glob
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras import backend as K


# =========================
# 기본 설정
# =========================

H5_DIR = "./models"
OUT_DIR = "./models_keras"
os.makedirs(OUT_DIR, exist_ok=True)

# 현재 학습 데이터 기준
x_train = np.load("./data/x_train.npy")
y_train = np.load("./data/y_train.npy")

vocab_size = 14999
max_len = x_train.shape[1]
num_classes = y_train.shape[1]


# =========================
# 파일명에서 모델 정보 추출
# 예:
# model_0.734043_model1_bat64_emb512.h5
# model_0.734043_model2_bat128_emb640_classW.h5
# =========================

def parse_filename(path):
    name = os.path.basename(path)

    m = re.search(r"_model(\d+)_bat(\d+)_emb(\d+)", name)
    if not m:
        raise ValueError("파일명에서 model/bat/emb 정보를 못 찾음: " + name)

    model_no = int(m.group(1))
    batch_size = int(m.group(2))
    embed_dim = int(m.group(3))
    use_class_weight = "classW" in name

    return name, model_no, batch_size, embed_dim, use_class_weight


# =========================
# 기존 model.add 구조와 같은 weight 구조를 Functional로 재현
# Lambda 없이 MaxPooling + AvgPooling + Concatenate 사용
# =========================

def build_model(model_no, embed_dim):
    inputs = Input(shape=(max_len,))

    x = Embedding(vocab_size, embed_dim)(inputs)

    if model_no == 1:
        x = SpatialDropout1D(0.30)(x)

        x = Conv1D(320, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Conv1D(256, 5, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Conv1D(192, 7, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Bidirectional(GRU(96, activation="tanh", return_sequences=True))(x)
        x = Dropout(0.35)(x)

        max_pool = GlobalMaxPooling1D()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        x = Concatenate()([max_pool, avg_pool])

        x = Dense(128, activation="relu")(x)
        x = Dropout(0.45)(x)

        x = Dense(128, activation="relu")(x)
        x = Dropout(0.35)(x)

    elif model_no == 2:
        x = SpatialDropout1D(0.32)(x)

        x = Conv1D(320, 3, padding="same", activation="relu",
                   kernel_regularizer=l2(5e-7))(x)
        x = BatchNormalization()(x)

        x = Conv1D(256, 5, padding="same", activation="relu",
                   kernel_regularizer=l2(8e-6))(x)
        x = BatchNormalization()(x)

        x = Conv1D(192, 7, padding="same", activation="relu",
                   kernel_regularizer=l2(3e-5))(x)
        x = BatchNormalization()(x)

        x = Bidirectional(GRU(96, activation="tanh", return_sequences=True))(x)
        x = Dropout(0.38)(x)

        max_pool = GlobalMaxPooling1D()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        x = Concatenate()([max_pool, avg_pool])

        x = Dense(128, activation="relu", kernel_regularizer=l2(8e-5))(x)
        x = Dropout(0.48)(x)

        x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.35)(x)

    elif model_no == 3:
        x = SpatialDropout1D(0.33)(x)

        x = Conv1D(352, 3, padding="same", activation="relu",
                   kernel_regularizer=l2(5e-7))(x)
        x = BatchNormalization()(x)

        x = Conv1D(288, 5, padding="same", activation="relu",
                   kernel_regularizer=l2(8e-6))(x)
        x = BatchNormalization()(x)

        x = Conv1D(192, 7, padding="same", activation="relu",
                   kernel_regularizer=l2(4e-5))(x)
        x = BatchNormalization()(x)

        x = Bidirectional(GRU(96, activation="tanh", return_sequences=True))(x)
        x = Dropout(0.40)(x)

        max_pool = GlobalMaxPooling1D()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        x = Concatenate()([max_pool, avg_pool])

        x = Dense(160, activation="relu", kernel_regularizer=l2(8e-5))(x)
        x = Dropout(0.48)(x)

        x = Dense(96, activation="relu", kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.35)(x)

    else:
        raise ValueError("model_no는 1, 2, 3만 가능")

    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)


# =========================
# 변환 실행
# =========================

ok = []
fail = []

for h5_path in sorted(glob.glob(os.path.join(H5_DIR, "model_*.h5"))):
    K.clear_session()

    try:
        name, model_no, batch_size, embed_dim, use_class_weight = parse_filename(h5_path)

        print("\n" + "=" * 100)
        print("CONVERT:", name)
        print(f"model={model_no}, batch={batch_size}, emb={embed_dim}, classW={use_class_weight}")

        model = build_model(model_no, embed_dim)

        # 핵심:
        # load_model()이 아니라 load_weights()만 사용
        model.load_weights(h5_path)

        out_name = name.replace(".h5", ".keras")
        out_path = os.path.join(OUT_DIR, out_name)

        model.save(out_path)

        print("OK:", out_path)
        ok.append((name, out_path))

    except Exception as e:
        print("FAIL:", os.path.basename(h5_path))
        print(type(e).__name__, e)
        fail.append((os.path.basename(h5_path), str(e)))


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print("성공:", len(ok))
print("실패:", len(fail))

if fail:
    print("\n실패 목록:")
    for name, err in fail:
        print(name, "|", err)