# %% Data Imbalance Problem
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

x, y = make_classification(n_samples=2000, n_features=6, weights=[0.95], flip_y=0)
print(Counter(y))

# %% 1. Under Sampling
# 1-1. Random Under Sampling
undersample = RandomUnderSampler(sampling_strategy='majority')
x_under, y_under = undersample.fit_resample(x, y)
print(Counter(y_under))

undersample = RandomUnderSampler(sampling_strategy=0.5) 
x_under2, y_under2 = undersample.fit_resample(x, y)
print(Counter(y_under2))

# 1-2. Tomek Links
# 1-3. Cluster Centroids
# 1-4. Near Miss
# 1-5. Condensed Nearest Neighbour
# 1-6. One Sided Selection
# 1-7. Edited Nearest Neighbours
# 1-8. Repeated Edited Nearest Neighbours
# 1-9. All KNN
# 1-10. Instance Hardness Threshold
# 1-11. Neighbourhood Cleaning Rule



# 2. Over Sampling

# 2-1. Random Over Sampling
# 2-2. SMOTE
# 2-3. ADASYN
# 2-4. Borderline SMOTE
# 2-5. Borderline SMOTE SVM
# 2-6. SVMSMOTE
# 2-7. KMeans SMOTE
# 2-8. Random SMOTE
# 2-9. SMOTEENN
# 2-10. SMOTETomek
# 2-11. GAN
# 2-12. VAE
# 2-13. WGAN
# 2-14. CGAN

# %%
