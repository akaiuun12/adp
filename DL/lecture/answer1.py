# %% [로지스틱 회귀와 시그모이드 함수 개념 교육용 파이썬 코드 예제]
# 로지스틱 회귀는 이진 분류 문제를 해결하는 데 사용되는 통계 모델입니다. 로지스틱 회귀는 시그모이드 함수를 사용하여 입력 데이터를 특정 클래스(0 또는 1)로 분류합니다. 

# 1. 라이브러리 임포트
# 2. 시그모이드 함수 정의
# 3. 데이터 생성
# 랜덤한 X 값을 생성하고, X가 0보다 크면 1, 작거나 같으면 0으로 라벨링하여 이진 분류 데이터를 만듭니다.
# 4. 초기 설정:
# 학습률 (learning_rate)와 반복 횟수 (n_iterations)를 설정합니다.
# 초기 가중치와 절편을 랜덤하게 설정합니다.
# 5. 절편 항 추가:
# X 데이터에 절편 항(1로 채워진 열)을 추가하여 X_b를 생성합니다. 이는 로지스틱 회귀 모델에서 절편을 포함하기 위함입니다.
# 6. 로지스틱 회귀 모델 학습 (경사 하강법):
# 경사 하강법을 사용하여 가중치(θ)를 최적화합니다. 각 반복마다 시그모이드 함수를 통해 예측을 수행하고, 그래디언트를 계산하여 가중치를 업데이트합니다.
# 7. 결과 출력
# 8. 예측 함수
# 9. 예측값 계산 및 시각화
 

import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))   #exp 함수는 자연 지수 함수 계산 
 

# 데이터 생성 (간단한 이진 분류 데이터)
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1  # -1과 1 사이의 값
y = (X > 0).astype(np.int64)  # X가 0보다 크면 1, 작거나 같으면 0

# 학습률과 반복 횟수 설정
learning_rate = 0.1
n_iterations = 1000

# 초기 가중치와 절편 설정
theta = np.random.randn(2, 1)

# X 데이터에 절편 항 추가 (1로 채워진 열 추가)
X_b = np.c_[np.ones((100, 1)), X]

# 로지스틱 회귀 모델 학습 (경사 하강법)
m = len(X_b)
for iteration in range(n_iterations):
    z = X_b.dot(theta)
    predictions = sigmoid(z)
    gradients = 1/m * X_b.T.dot(predictions - y)
    theta -= learning_rate * gradients

# 결과 출력
print("최종 가중치 (theta):", theta)

# 예측 함수
def predict_proba(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]  # 절편 항 추가
    return sigmoid(X_b.dot(theta))

def predict(X, theta):
    return (predict_proba(X, theta) >= 0.5).astype(np.int64)

# 예측값
y_pred = predict(X, theta)

# 데이터와 결정 경계를 시각화
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, predict_proba(X, theta), color='red', label='Sigmoid function')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

