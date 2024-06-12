# %% Linear Regression using TensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %% 데이터 생성
np.random.seed(42)

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95])

# 모델 구성
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, activation='linear')
# ])


inputs = tf.keras.layers.Input(shape=(1,))
output = tf.keras.layers.Dense(1, activation='linear')(inputs)
linear_model = tf.keras.models.Model(inputs, output)

sgd = tf.keras.optimizers.SGD(lr=0.01)

# 모델 설정
linear_model.compile(loss='mean_squared_error', 
                     optimizer=sgd)

# 모델 학습
history = linear_model.fit(X, y, epochs=1000, verbose=0)

# 결과 출력
plt.scatter(X, y)
plt.plot(X, linear_model.predict(X), 'r')

plt.show()