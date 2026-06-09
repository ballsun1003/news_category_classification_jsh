import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.losses import CategoricalCrossentropy
from keras import backend as K

from sklearn.utils.class_weight import compute_class_weight


# =========================
# 기본 설정
# =========================

# 데이터 로드
x_train = np.load('./data/x_train.npy')
y_train = np.load('./data/y_train.npy')
x_test = np.load('./data/x_test.npy')
y_test = np.load('./data/y_test.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 토크나이저 기준 단어장 크기
# tokenizer.word_index 길이 + 1 값으로 맞춰야 함
vocab_size = 14999

# 입력 길이 / 클래스 수 자동 감지
max_len = x_train.shape[1]
num_classes = y_train.shape[1]

# 저장 폴더
SAVE_DIR = './models'
os.makedirs(SAVE_DIR, exist_ok=True)

# 결과 CSV
RESULT_CSV = './training_results.csv'

# 54개 전부 플롯 띄우면 귀찮으니까 기본 False
PLOT_EACH = False

# 실험 조합
MODEL_NUMBERS = [1, 2, 3]
BATCH_SIZES = [64, 128, 192]
EMBED_DIMS = [512, 640, 768]
USE_CLASS_WEIGHT_LIST = [False, True]


# =========================
# GPU 메모리 설정
# =========================
# 여러 모델을 연속 학습할 때 GPU 메모리 점유 문제 줄이기용
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


# =========================
# class_weight 계산
# =========================
# y_train이 one-hot이므로 정수 라벨로 변환
y_train_int = np.argmax(y_train, axis=1)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)

class_weight_dict = dict(enumerate(weights))

print('class_weight:', class_weight_dict)


# =========================
# 모델 생성 함수
# =========================
def build_model(model_no, embed_dim):
    """
    model_no:
        1 = 기존 최고 모델 구조 거의 그대로
        2 = 보수적으로 정규화/드롭아웃 살짝 추가
        3 = 데이터 증가 반영해서 Conv/Dense 살짝 확장

    embed_dim:
        Embedding 차원. 512 / 640 / 768 실험용.
    """

    model = Sequential()

    # 단어 임베딩
    # vocab_size: 단어장 크기
    # embed_dim: 단어 하나를 몇 차원 벡터로 표현할지
    model.add(Embedding(vocab_size, embed_dim))
    model.build(input_shape=(None, max_len))

    # 임베딩 과적합 방지
    if model_no == 1:
        model.add(SpatialDropout1D(0.30))
    elif model_no == 2:
        model.add(SpatialDropout1D(0.32))
    elif model_no == 3:
        model.add(SpatialDropout1D(0.33))

    # =========================
    # Conv 블록
    # =========================
    # kernel 3: 짧은 핵심 키워드 조합
    # kernel 5: 중간 길이 구문
    # kernel 7: 긴 제목 패턴
    # 긴 커널일수록 외울 위험이 커서 model 2/3에서는 L2를 더 강하게 줌

    if model_no == 1:
        # 기존 최고 모델과 최대한 동일한 구조
        model.add(Conv1D(320, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())

        model.add(Conv1D(256, 5, padding='same', activation='relu'))
        model.add(BatchNormalization())

        model.add(Conv1D(192, 7, padding='same', activation='relu'))
        model.add(BatchNormalization())

        gru_units = 96
        gru_dropout = 0.35

        dense1_units = 128
        dense2_units = 128
        dense1_dropout = 0.45
        dense2_dropout = 0.35

        dense1_reg = None
        dense2_reg = None

    elif model_no == 2:
        # 보수 튜닝 버전
        # 구조는 거의 유지하고, 과적합 방지용 L2/Dropout만 살짝 추가
        model.add(Conv1D(320, 3, padding='same', activation='relu',
                         kernel_regularizer=l2(5e-7)))
        model.add(BatchNormalization())

        model.add(Conv1D(256, 5, padding='same', activation='relu',
                         kernel_regularizer=l2(8e-6)))
        model.add(BatchNormalization())

        model.add(Conv1D(192, 7, padding='same', activation='relu',
                         kernel_regularizer=l2(3e-5)))
        model.add(BatchNormalization())

        gru_units = 96
        gru_dropout = 0.38

        dense1_units = 128
        dense2_units = 128
        dense1_dropout = 0.48
        dense2_dropout = 0.35

        dense1_reg = l2(8e-5)
        dense2_reg = l2(1e-4)

    elif model_no == 3:
        # 데이터 증가 반영 확장 버전
        # Conv 3/5를 살짝 키우고 Dense 구성을 160 -> 96으로 조정
        model.add(Conv1D(352, 3, padding='same', activation='relu',
                         kernel_regularizer=l2(5e-7)))
        model.add(BatchNormalization())

        model.add(Conv1D(288, 5, padding='same', activation='relu',
                         kernel_regularizer=l2(8e-6)))
        model.add(BatchNormalization())

        model.add(Conv1D(192, 7, padding='same', activation='relu',
                         kernel_regularizer=l2(4e-5)))
        model.add(BatchNormalization())

        gru_units = 96
        gru_dropout = 0.40

        dense1_units = 160
        dense2_units = 96
        dense1_dropout = 0.48
        dense2_dropout = 0.35

        dense1_reg = l2(8e-5)
        dense2_reg = l2(1e-4)

    else:
        raise ValueError('model_no must be 1, 2, or 3')

    # =========================
    # BiGRU 블록
    # =========================
    # return_sequences=True:
    # 각 토큰 위치별 출력 유지.
    # 그래야 뒤에서 MaxPooling / AveragePooling 가능.
    model.add(Bidirectional(GRU(gru_units, activation='tanh', return_sequences=True)))
    model.add(Dropout(gru_dropout))

    # =========================
    # Max + Average pooling
    # =========================
    # Sequential에서는 GlobalMaxPooling1D와 GlobalAveragePooling1D를
    # 직접 concat하기 어려워서 Lambda로 처리.
    #
    # reduce_max  = 강한 키워드/패턴 하나 잡기
    # reduce_mean = 제목 전체 분위기 반영
    model.add(Lambda(lambda x: tf.concat([
        tf.reduce_max(x, axis=1),
        tf.reduce_mean(x, axis=1)
    ], axis=1)))

    # =========================
    # Dense 분류기
    # =========================
    model.add(Dense(dense1_units, activation='relu', kernel_regularizer=dense1_reg))
    model.add(Dropout(dense1_dropout))

    model.add(Dense(dense2_units, activation='relu', kernel_regularizer=dense2_reg))
    model.add(Dropout(dense2_dropout))

    model.add(Dense(num_classes, activation='softmax'))

    return model


# =========================
# 그래프 출력 함수
# =========================
def plot_history(history, title=''):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Trajectory ' + title)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Trajectory ' + title)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =========================
# 단일 실험 실행 함수
# =========================
def train_one(model_no, batch_size, embed_dim, use_class_weight):
    """
    한 조합 학습:
        model_no
        batch_size
        embed_dim
        use_class_weight

    학습 끝나면:
        score[1] 기준으로 파일명 생성
        모델 저장
        결과 dict 반환
    """

    # 이전 모델 그래프/메모리 정리
    K.clear_session()
    gc.collect()

    print('\n' + '=' * 100)
    print(f'MODEL {model_no} | batch={batch_size} | emb={embed_dim} | class_weight={use_class_weight}')
    print('=' * 100)

    model = build_model(model_no, embed_dim)
    model.summary()

    # 콜백은 모델마다 새로 만들어야 상태가 안 섞임
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            mode='max',
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # label_smoothing:
    # 라벨을 너무 100% 확신하지 않게 만들어 과적합 완화
    model.compile(
        loss=CategoricalCrossentropy(label_smoothing=0.02),
        optimizer=AdamW(learning_rate=2e-4, weight_decay=1e-5),
        metrics=['accuracy']
    )

    # class_weight 사용 여부
    # True면 소수 클래스 loss를 더 크게 반영
    cw = class_weight_dict if use_class_weight else None

    fit_hist = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1000,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    loss = float(score[0])
    acc = float(score[1])

    print(f'FINAL SCORE | loss={loss:.6f} | acc={acc:.6f}')

    if PLOT_EACH:
        title = f'm{model_no}_bat{batch_size}_emb{embed_dim}_cw{use_class_weight}'
        plot_history(fit_hist, title)

    # 파일명 생성
    # class_weight를 쓴 경우에만 _classW 붙임
    cw_tag = '_classW' if use_class_weight else ''
    filename = f'model_{acc:.6f}_model{model_no}_bat{batch_size}_emb{embed_dim}{cw_tag}.h5'
    save_path = os.path.join(SAVE_DIR, filename)

    model.save(save_path)
    print('saved:', save_path)

    # 결과 저장용 정보
    result = {
        'model_no': model_no,
        'batch_size': batch_size,
        'embed_dim': embed_dim,
        'class_weight': use_class_weight,
        'loss': loss,
        'accuracy': acc,
        'best_val_accuracy_in_history': float(max(fit_hist.history['val_accuracy'])),
        'best_val_loss_in_history': float(min(fit_hist.history['val_loss'])),
        'epochs_ran': len(fit_hist.history['loss']),
        'saved_path': save_path,
    }

    return result


# =========================
# 전체 조합 실행
# =========================

results = []

for model_no in MODEL_NUMBERS:
    for batch_size in BATCH_SIZES:
        for embed_dim in EMBED_DIMS:
            for use_class_weight in USE_CLASS_WEIGHT_LIST:
                try:
                    result = train_one(
                        model_no=model_no,
                        batch_size=batch_size,
                        embed_dim=embed_dim,
                        use_class_weight=use_class_weight
                    )

                    results.append(result)

                    # 매번 CSV 저장
                    # 중간에 터져도 이전 결과는 남음
                    pd.DataFrame(results).to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')

                except Exception as e:
                    print('\nERROR!')
                    print(f'model_no={model_no}, batch={batch_size}, emb={embed_dim}, class_weight={use_class_weight}')
                    print(e)

                    # 에러도 CSV에 남김
                    error_result = {
                        'model_no': model_no,
                        'batch_size': batch_size,
                        'embed_dim': embed_dim,
                        'class_weight': use_class_weight,
                        'loss': None,
                        'accuracy': None,
                        'best_val_accuracy_in_history': None,
                        'best_val_loss_in_history': None,
                        'epochs_ran': None,
                        'saved_path': None,
                        'error': str(e),
                    }

                    results.append(error_result)
                    pd.DataFrame(results).to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')

                    # 다음 조합 계속 진행
                    K.clear_session()
                    gc.collect()


# =========================
# 최종 순위 출력
# =========================

df_result = pd.DataFrame(results)

# 성공한 결과만 정렬
df_ok = df_result.dropna(subset=['accuracy']).sort_values('accuracy', ascending=False)

print('\n' + '=' * 100)
print('FINAL RANKING')
print('=' * 100)
print(df_ok[['accuracy', 'loss', 'model_no', 'batch_size', 'embed_dim', 'class_weight', 'saved_path']])

df_result.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
print('\nresult saved:', RESULT_CSV)