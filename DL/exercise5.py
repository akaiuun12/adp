# %% 

# [와인 데이터 이진분류 신경망 학습]
# 1. Scikit-learn의 와인 데이터셋을 로드합니다.
# 2. 데이터 정규화를 위해 StandardScaler를 사용합니다.
# 3. 타겟 레이블을 원-핫 인코딩합니다.
# 4.학습 데이터와 테스트 데이터를 분리합니다.
# 5. Keras를 사용하여 신경망 모델을 구성합니다.
# 6. 모델을 컴파일하고 학습시킵니다.
# 7. 테스트 데이터로 모델을 평가합니다.

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.utils import to_categorical

# 와인 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 타겟 원-핫 인코딩
y_categorical = to_categorical(y)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
