# %% 제4회 기출동형 모의고사 - 통계분석 (50점)

# %% 1. 한공장에서 생산된 제품에서 최근 추정 불량률은 90%였다. 오차의 한계가 5% 이하가 되도록 하는 최소 표본 사이즈를 구하시오.


# %% 2. 다음은 1월부터 9월까지의 은의 가격이다. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Batang')

date = [1,2,3,4,5,6,7,8,9]
value = [12.14, 42.6, 34.4, 35.29, 30.96, 57.12, 37.84, 42.49, 31.38]

df = pd.DataFrame(value, index=date)
df

# 2-1. 은의 가격 및 이동평균값 3이 설정된 시계열 그래프를 그리시오. 
ma3 = []
for i in range(df.shape[0]):
    if i < 2:
        ma3.append(0)
    else:
        ma3.append(np.mean(df.iloc[i-2:i][0]))
df['MA3'] = ma3

plt.plot(df)
plt.title('은 가격 (1월 ~ 9월)')
plt.show()

# 2-2. 1월 대비 9월의 은의 가격은 몇 %올랐는가? 소수점 두번째 자리에서 반올림.
rate = float(df.loc[9][0] / df.loc[1][0] - 1)
print(f'1월 대비 9월 은의 가격은 {rate * 100: .1f}% 상승함.')


# %% 3. 아래 그래프는 A, B, C 자치구별 H의원에 대한 찬성, 반대 투표 결과이다. 자치구별 지지율이 같은지에 대해서 검정하시오.


# %% 4. A학교 남녀 학생들의 평균 혈압 차이가 있는지 여부에 대한 검정하시오. 
# 4-1. 남녀 학생들의 평균 혈압 차이가 있는지에 대해 가설을 설정하시오.
# A: 
# H0 - 남녀 학생들의 평균 혈압 차이가 존재할 것이다.
# H1 - 남녀 학생들의 평균 혈압 차이가 존재하지 않을 것이다. 

# 4-2. 검정통계량을 구하고 판단하시오.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

df = pd.read_csv('../../ADP_Python/data/26_problem6.csv')
dv = 'pressure'
iv = 'gender'
conditions = {'A':'male', 'B':'female'}

group_A = df[df[iv] == conditions['A']]
group_B = df[df[iv] == conditions['B']]

# 3-2. Visualization
fig, axes = plt.subplots(nrows=2, ncols=1)

sns.histplot(group_A[dv], bins=15, ax=axes[0])
sns.histplot(group_B[dv], bins=15, ax=axes[1])
plt.tight_layout
plt.show()

# 3-3. Statistical Test
print(stats.ttest_ind(group_A[dv], group_B[dv], equal_var=True))

print(f'Group A Mean: {group_A[dv].mean()}')
print(f'Group B Mean: {group_B[dv].mean()}')

# Welch's t-test를 수행한 결과 p-value = 0.19로 귀무가설을 기각할 수 없다.
# 따라서 남녀 학생들의 평균 혈압 차이는 통계적으로 유의미하지 않다.


# 4-3. 검정통계량을 구하고 판단하시오.
import statsmodels.formula.api as smf

model = smf.ols(
    data=df, formula=f'{dv} ~ 1 + {iv}'
).fit()

print(model.summary())

# 해당 데이터의 신뢰구간은 [-2.635, 13.228]이다. 
# 0값이 신뢰구간 내에 포함되기 때문에 두 집단의 차이가 통계적으로 유의미하다는 결론을 내릴 수 없으며
# 이는 검정통계량을 사용한 판단과 일치한다. 

# %% 5. height(키), weight(몸무게), waist(허리둘레) 컬럼을 가진 problem7.csv 파일을 가지고 다음을 분석하시고.
# A시의 20대 남성 411명을 임의로 추출한 후 키, 몸무게, 허리둘레를 조사하여 기록한 데이터이다. 
# 이 데이터를 이용하여 20대 남성의 키와 허리둘레가 체중에 영향을 미치는지 알아보시오.

# 5-1. 아래 조건을 참고하여 회귀 계수 (반올림하여 소수점 두자리)를 구하시오.

# 5-2. 위에서 만든 모델을 바탕으로 키 180cm, 허리둘레 85cm인 남성의 몸무게를 추정하시오.