import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *


x_train = np.load('./data/x_train.npy')
y_train = np.load('./data/y_train.npy')
x_test = np.load('./data/x_test.npy')
y_test = np.load('./data/y_test.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

vocab_size = int(x_train.max()) + 1
max_len = x_train.shape[1]
num_classes = y_train.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 512))
model.build(input_shape=(None, max_len))

model.add(SpatialDropout1D(0.28))

model.add(Conv1D(384, 3, padding='same', kernel_regularizer=l2(5e-7)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(320, 5, padding='same', kernel_regularizer=l2(8e-6)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(224, 7, padding='same', kernel_regularizer=l2(3e-5)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.18))

model.add(Bidirectional(GRU(80, activation='tanh', return_sequences=True)))
model.add(Dropout(0.38))

model.add(GlobalMaxPooling1D())

model.add(Dense(192, kernel_regularizer=l2(8e-5)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.45))

model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
model.add(Dropout(0.30))

model.add(Dense(num_classes, activation='softmax'))
model.summary()

callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1,
    ),
]

model.compile(loss='categorical_crossentropy', optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5), metrics=['accuracy'])
fit_hist = model.fit(x_train,y_train,batch_size=128,epochs=1000,validation_data=(x_test,y_test), callbacks=callbacks, verbose=1)
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
model.save('./model_{}.h5'.format(score[1]))