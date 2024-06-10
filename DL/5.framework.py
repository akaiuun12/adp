# %% 0.필요한 패키지 불러오기
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# %% 1. 데이터셋 생성하기
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# %% 2. 모델 구성하기
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %% 3. 모델 실행 옵션을 설정하기
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# 모델 최적화를 위한 설정 구간입니다.
modelpath = "./MNIST_MLP.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


# %% 4. 모델 학습시키기
history = model.fit(X_train, y_train, 
                    validation_split=0.25, epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])

# 테스트 정확도를 출력합니다.
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

# 손실값과 검증 손실값 저장
y_loss = history.history['loss']
y_vloss = history.history['val_loss']


# %% 5. 학습과정 살펴보기 (그래프로 표현)
lenloss = np.arange(len(y_loss))
plt.plot(lenloss, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(lenloss, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# %% 6. 모델 평가하기
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
print("## evaluation loss and_metrics ##")
print(loss_and_metrics)


# %% 7. 모델 사용하기
xhat = X_test[0:1]
yhat = model.predict(xhat)
print("## yhat ##")
print(yhat)