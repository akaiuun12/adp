# %% 제4회 기출동형 모의고사 - 머신러닝
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../../ADP_Python/data/26_problem1.csv')

print(df.info())

# %% 1. 데이터 전처리 및 군집생성
# 1-1. 결측치를 확인하고 결측치를 제거하시오.
print(df.isna().sum())
df['Income'].fillna(df['Income'].median(), inplace=True)

# 총 2240개 데이터 중 24개 데이터의 Income 값에 결측치가 존재한다.
# 결측치를 처리하기 위해서는 결측치가 포함된 데이터를 삭제하는 방법과, 다른 값으로 대체하는 방법이 있다. 
# 본 분석에서는 Income 변수의 특성을 고려하여 중간값으로 대체하는 방식으로 결측치를 제거하였다.

# 1-2. 이상치를 제거하는 방법을 서술하고 이상치 제거 후 결과를 통계적으로 나타내시오.
sns.boxplot(df.drop(['ID', 'Income', 'Marital_Status'], axis=1))
plt.show()

# 1-3. 위에서 전처리한 데이터로 Kmeans, DBSCAN 등의 방법으로 군집을 생성하시오.


# %% 2. 군집분석
# 2-1. 위에서 생성한 군집들의 특성을 분석하시오. 

# 2-2. 각 군집별 상품을 추천하시오.

# 2-3. ID가 10870인 고객을 대상으로 상품을 추천하시오.