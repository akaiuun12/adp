# %% sample_test_1_ml.py


# %% 1.

# %% 2. 코로나 시계열 데이터로 다음을 수행하시오
# %% 0. Load Libraries and Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

# data encoding type: 'utf-8', 'euc-kr'
df = pd.read_csv('../../ADP_Python/data/서울특별시 코로나19.csv')
date_column = '날짜'

# Check data properties
print(df.head())
print(df.info())

# # Change datatype to datetime
df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d')
df.set_index(date_column, inplace=True)

print(df.dtypes)
print(df.head())

# EDA Visualization
plt.plot(df)
plt.show()


# %% 1. Time Series Decomposition 
# (Trend, Seasonality, Residual)
# - 'additive'
# - 'multiplicative'

# from statsmodels.tsa.seasonal import seasonal_decompose

# decomp_add = seasonal_decompose(df, model='additive')
# decomp_mul = seasonal_decompose(df, model='multiplicative')

# decomp_add.plot()
# decomp_mul.plot()
# plt.show()


# %% 2. Stationarize the Series
# %% 2-1. Durbin-Watson Test
from statsmodels.stats.stattools import durbin_watson

print(durbin_watson(df))

# %% 2-2. Augmented Dickey-Fuller Test (d)
# Stationary Test
from statsmodels.tsa.stattools import adfuller

# train, test data split
df_train = df[:'2016-12-01']
df_test = df.drop(df_train.index)

print(df_train)
print(df_test)

adf = adfuller(df_train, regression='ct')

print(f'ADF Statistic: {adf[0]}')
print(f'p-value: {adf[1]}')

if adf[1] < 0.05:
    print('stationary time-series data')
else:
    print('WARNING: non-stationary time-serie data')
    print('WARNING: differentiation or log transformation needed')

# %% 2-3. Differentiation
# First-order differentiation
df_diff1 = df.diff(1)
df_diff1 = df_diff1.dropna()

df_diff1.plot()
plt.show()

adf1 = adfuller(df_diff1)

print(f'ADF Statistic: {adf1[0]}')
print(f'p-value: {adf1[1]}')

# Second-order differentiation
df_diff2 = df.diff(2)
df_diff2 = df_diff2.dropna()

df_diff2.plot()
plt.show()

adf2 = adfuller(df_diff2)

print(f'ADF Statistic: {adf2[0]}')
print(f'p-value: {adf2[1]}')

# 2-3. Log Transformation
# 2-4. Box-Cos Transformation


# %% 3. Plot ACF/PACF Charts and Find Optimal Parameters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %% 3-1. AR (Auto Regressive) Model: AR(p)
# PACF (p)
plot_pacf(df_diff1)
plt.show()

print(pacf(df_diff1))

# %% 3-2. MA (Moving Average) Model: MA(q)
# ACF (q)
plot_acf(df_diff1)
plt.show()

print(acf(df_diff1))