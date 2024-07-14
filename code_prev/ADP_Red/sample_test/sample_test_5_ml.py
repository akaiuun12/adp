# %% sample_test_5_ml.py
# %% 0. 패키지 및 데이터 불러오기
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

df = pd.read_csv('../../ADP_Python/data/27_problem1.csv',
                 encoding='euc-kr')
dv = 'Class'

X = df.drop(dv, axis=1)
y = df[dv]

print(df.head())
print(df.info())


# %% 1. 데이터 전처리\
# 1-1. 데이터의 특징을 파악하시오 (EDA)
print(X.info())
sns.boxplot(data=X.drop(['Time', 'Amount'], axis=1))
plt.show()

# 총 1193 데이터가 존재한다. 모든 데이터는 float64 형식이며 결측치는 없다. 
# Time 및 Amount 변수를 제외하고도 IQR 기준으로 이상치가 다수 존재한다.

# 1-2. 상관관계를 시각화하고 전처리가 필요함을 설명하시오.
X_corr = X.corr(method='pearson')
sns.heatmap(abs(X_corr) > 0.5)
plt.tight_layout()
plt.show()

# 총 18개(V1~V17 + Amount)의 독립 변수를 확인한 결과, 변수 간 높은 상관성이 나타나 다중공선성이 의심된다. 
# 총 70개 독립 변수 쌍의 상관관계의 절대값이 0.5 이상이었다.
# 다중공선성은 모형의 정확도를 낮출 수 있기 때문에, 다중공선성을 해결하기 위해 차원 축소를 시행하도록 한다.


# %% 2. 차원 축소
# 2-1. 차원 축소 방법 2가지 이상을 비교하고 한 가지를 선택하시오.
# 1. 변수선택법
# EDA에서 상관관계가 매우 높았던 설명변수는 유사한 정보를 가지고 있을 것이라고 판ㅏㄴ하고 그 중 하나만을 선택해 분석에 사용할 수 있다. 이러한 설명 변수 선택의 장점은 선택한 설명변수의 해석이 용이하고 수행 과정이 간단하다는 것이다. 하지만 설명변수간의 고차원적인 상관관계는 고려하기 어렵다는 단점이 있다. 

# 2. PCA
# 따라서 차원 축소를 위해 주성분 분석을 사용하도록 한다. 주성분 분석은 기존의 컬럼을 새롭게 해석하여 저차원의 초평면에 투영한다.

# 2-2. 위서 선택한 방법을 실제로 수행하고 선택한 이유를 설명하시오.

# PCA를 사용하기 위해 데이터를 표준화한다.
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

# Scree Plot을 그려서 주성분의 개수를 선택한다.            
from sklearn.decomposition import PCA

pca = PCA(n_components=10).fit(X_scaled)

plt.plot(pca.explained_variance_ratio_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# PCA를 통해 2개의 주성분을 선택한다.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Class'] = y

# PCA를 통해 차원을 축소한 데이터를 시각화한다.
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Class')
plt.title('PCA')
plt.show()


# %% 3. 오버샘플링과 언더샘플링

# %% 4. 이상탐지 모델