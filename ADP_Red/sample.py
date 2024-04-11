# %% 0. Import Libaries
# !conda install numpy
# !conda install pandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

# %% 1. One-sample t-test
df = pd.read_csv('../ADP_Python/data/cats.csv')
dv = 'Bwt'

print(f'{df.info()}')
dv_mean = df[dv].mean()
pvalue = stats.ttest_1samp(df[dv], popmean=2.6)[1]

print(f'''
df 데이터는 총 {df.shape[0]}개의 데이터로 구성되어 있다.
{dv} 데이터 평균은 {dv_mean:.2f}이며 
one-sample t-test 결과
p-value = {pvalue:.2f}으로 통계적으로 유의미하다.
''')


# 1-2. Data Visualization
fig, axes = plt.subplots(nrows=2, ncols=1)

sns.histplot(df[dv], bins=15, ax=axes[0])
sns.barplot(data=df, y=dv, ax=axes[1])
plt.tight_layout
plt.show()


# 1-3. Statistical Test
mu = 2.6

# 정규성 검증: Shapiro-Wilks test
# Shapiro-Wilks test가 p-value < 0.05로 기각된다면
# 데이터가 정규분포를 따르지 않는다는 뜻이다.
val, pval = stats.shapiro(df[dv])

if pval > 0.05:
    # 데이터가 정규분포를 따를 경우 평범하게 one sample t-test를 진행하면 된다. 
    print(f'p-value:{pval:.2f}, Normality Assumption Satisfied!')
    print(f'Try Simple t-test!')
    print(stats.ttest_1samp(df[dv], popmean=mu))
else:
    # 데이터가 정규분포를 따르지 않을 경우 비모수 검정인 Wilcoxon test를 사용한다.
    # Wilcoxon test가 p-value < 0.05로 기각된다면     
    print(f'p-value:{pval:.2f}, Normality Assumption Unsatisfied!')
    print(f'Try Wilcoxon Test!')
    print(stats.wilcoxon(df[dv]-mu, alternative='two-sided'))

# 어느 쪽이든 고양이의 몸무게는 2.6이 아니라는 결과가 나오지만
# one sample t-test가 아닌 wilcoxon test를 사용해야 한다.


# %% 2. paired-sample t-test
data = {'before':[7,3,4,5,2,1,6,6,5,4],
       'after':[8,4,5,6,2,3,6,8,6,5]}

df = pd.DataFrame(data)
dv = 'diff'
df[dv] = df.after - df.before

# 2-2. Visualization
fig, axes = plt.subplots(nrows=2, ncols=1)

sns.histplot(df[dv], bins=15, ax=axes[0])
sns.barplot(data=df, y=dv, ax=axes[1])
plt.tight_layout
plt.show()

# 2-3. Statistical Test
val, pval = stats.shapiro(df[dv])

if pval < 0.05: # Normality assumption not satisfied
    print(f'p-value:{pval:.2f}, Normality Assumption Unsatisfied!')
    print(f'Try Wilcoxon Test!')
else: #
    print(f'p-value:{pval:.2f}, Normality Assumption Satisfied!')
    print(f'Try Simple t-test!')

# print(stats.wilcoxon(df.before, df.after, alternative='two-sided'))
print(stats.ttest_rel(df.after, df.before, alternative='greater'))

# %% 3. independent-sample t-test
# Independent sample t-test를 시행할 때 
# 등분산성 확인을 위해 Levene test 등을 수행할 필요는 없다.
# 등분산성 가정을 만족하든, 만족하지 않든 Welch's t-test를 수행하면 된다.
# 단, 정규성 조건은 여전히 유효하다.

# [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)

df = pd.read_csv('../ADP_Python/data/cats.csv')
dv = 'Bwt'
iv = 'Sex'
conditions = {'A':'M', 'B':'F'}

group_A = df[df[iv] == conditions['A']]
group_B = df[df[iv] == conditions['B']]

# 3-2. Visualization
fig, axes = plt.subplots(nrows=2, ncols=1)

sns.histplot(group_A[dv], bins=15, ax=axes[0])
sns.histplot(group_B[dv], bins=15, ax=axes[1])
plt.tight_layout
plt.show()

# 3-3. Statistical Test
print(stats.ttest_ind(group_A[dv], group_B[dv], equal_var=False))  # Welch's t-test

print(f'Group A Mean: {group_A[dv].mean()}')
print(f'Group B Mean: {group_B[dv].mean()}')