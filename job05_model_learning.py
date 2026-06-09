import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
from sklearn.utils.class_weight import compute_class_weight
from keras.losses import CategoricalCrossentropy

import tensorflow as tf


x_train = np.load('./data/x_train.npy')
y_train = np.load('./data/y_train.npy')
x_test = np.load('./data/x_test.npy')
y_test = np.load('./data/y_test.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

vocab_size = 14999
max_len = x_train.shape[1]
num_classes = y_train.shape[1]

inputs = Input(shape=(max_len,))

x = Embedding(vocab_size, 512)(inputs)
x = SpatialDropout1D(0.30)(x)

x = Conv1D(320, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv1D(256, 5, padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv1D(192, 7, padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Bidirectional(GRU(96, activation='tanh', return_sequences=True))(x)
x = Dropout(0.35)(x)

max_pool = GlobalMaxPooling1D()(x)
avg_pool = GlobalAveragePooling1D()(x)
x = Concatenate()([max_pool, avg_pool])

x = Dense(128, activation='relu')(x)
x = Dropout(0.45)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.35)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.summary()

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
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1,
    ),
]

y_int = np.argmax(y_train, axis=1)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_int),
    y=y_int
)

class_weight = dict(enumerate(weights))


model.compile(loss=CategoricalCrossentropy(label_smoothing=0.02), optimizer=AdamW(learning_rate=2e-4, weight_decay=1e-5), metrics=['accuracy'])
fit_hist = model.fit(x_train,y_train,batch_size=64,epochs=1000,validation_data=(x_test,y_test), callbacks=callbacks, class_weight=class_weight, verbose=1)
score = model.evaluate(x_test,y_test,verbose=0)


def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Trajectory')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Trajectory')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_history(fit_hist)
print(score)
model.save('./model_{}_bat64_emb512_classW.h5'.format(score[1]))