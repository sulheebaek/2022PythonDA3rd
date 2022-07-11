# keras의 VGG16 아키텍쳐를 이용한 전이학습 코드

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# seed 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

train_dir = os.path.join('Grape','train')
test_dir = os.path.join('Grape', 'test')
WIDTH, HEIGHT, CHANNELS = 256, 256, 3

# 이미지 전처리 함수 정의
def imagePrep(dir_, HEIGHT, WIDTH):
    dirs = os.listdir(dir_)
    
    X_train = []
    y_train = []
    
    for i, d in enumerate(dirs):
        for fname in os.listdir(os.path.join(dir_, d)):
            fpath = os.path.join(dir_, d, fname)
            img = image.load_img(fpath, target_size=(HEIGHT, WIDTH)) # 이미지 읽어들이며 리사이즈
            img_ = image.img_to_array(img) # PIL Image객체를 np array로 변환
            X_train.append(img_/255.0)
            y_train.append(i)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = to_categorical(y_train)
    return X_train, y_train

def createModel():
    # VGG16을 이용한 전이학습 모델 준비

    transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))
    transfer_model.trainable = False
    transfer_model.summary()

    finetune_model = Sequential()
    finetune_model.add(transfer_model)
    finetune_model.add(Flatten())
    finetune_model.add(Dense(64, activation='relu'))
    finetune_model.add(Dense(4, activation='softmax'))
    finetune_model.summary()

    finetune_model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate=0.0002), 
                      metrics='accuracy')
    return finetune_model

if __name__ == '__main__':

    X_train, y_train = imagePrep(train_dir, HEIGHT, WIDTH)
    X_test, y_test = imagePrep(test_dir, HEIGHT, WIDTH)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    finetune_model = createModel()
    
    # 모델 최적화 설정
    MODEL_DIR = './finetune_model/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    modelpath="./finetune_model/{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=modelpath, 
                                                      monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # 모델의 실행
    history = finetune_model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=3, batch_size=5, verbose=0, 
                        callbacks=[early_stopping_callback, checkpointer])

    # 테스트 정확도 출력
    print('\n Test Accurary: %.4f' % (finetune_model.evaluate(X_test, y_test)[1]))

    # 테스트셋의 오차
    y_vloss = history.history['val_loss']

    # 학습셋의 오차
    y_loss = history.history['loss']

    # 시각화
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()