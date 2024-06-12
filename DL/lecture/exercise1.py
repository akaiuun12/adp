# %% Logistic Classification using TensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %% 데이터 생성
np.random.seed(42)

X = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0,0,0,1,1,1,1])

# 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 설정
model.compile(loss='binary_crossentropy', 
              optimizer='sgd', metrics=['accuracy'])

# 모델 학습
history = model.fit(X, y, epochs=100)

# 결과 출력
plt.scatter(X, y)
plt.plot(X, model.predict(X), 'r')
plt.show()

hour = 7
prediction = model.predict([hour])
print('7시간 공부했을때 합격 확률 ' , prediction*100)