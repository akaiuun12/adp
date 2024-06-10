# %% 
import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))   #exp 함수는 자연 지수 함수 계산 
 

# 데이터 생성 (간단한 이진 분류 데이터)
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1  # -1과 1 사이의 값
y = (X > 0).astype(np.int)  # X가 0보다 크면 1, 작거나 같으면 0

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
    return (predict_proba(X, theta) >= 0.5).astype(np.int)

# 예측값
y_pred = predict(X, theta)

# 데이터와 결정 경계를 시각화
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, predict_proba(X, theta), color='red', label='Sigmoid function')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()




# %%
