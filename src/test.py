import os
import sys
import warnings
from pathlib import Path
from time import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from scipy.stats import spearmanr
from utils import MultipleTimeSeriesCV

warnings.filterwarnings("ignore")

np.random.seed(42)

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import MultipleTimeSeriesCV

scope_params = ["lookahead", "train_length", "test_length"]
daily_ic_metrics = [
    "daily_ic_mean",
    "daily_ic_mean_n",
    "daily_ic_median",
    "daily_ic_median_n",
]
lgb_train_params = [
    "learning_rate",
    "num_leaves",
    "feature_fraction",
    "min_data_in_leaf",
]
rf_train_params = [
    "bagging_fraction",
    "feature_fraction",
    "min_data_in_leaf",
    "max_depth",
]

idx = pd.IndexSlice

# data = pd.read_hdf('data.h5', 'model_data').sort_index()
data = pd.read_hdf("data/assets.h5", "engineered_features").sort_index()  # modificado

# labels = sorted(data.filter(like='_fwd').columns)
labels = sorted(data.filter(like="target").columns)
features = data.columns.difference(labels).tolist()
# label = f'r{lookahead:02}_fwd'
label = "target_1w"  # modificado

# Encuentra las filas con al menos un valor NaN
nan_cols = data.loc[idx[:, "2024":], features + [label]].isna().any(axis=0)

print(nan_cols[nan_cols == True])

# completamos con los valores del periodo anterior, para evitar que el último dato apareza nan
data = data.fillna(method="ffill")

# datos desde 2010
data = data.loc[idx[:, "2010":], features + [label]].dropna()

categoricals = ["month", "sector"]

for feature in categoricals:
    data[feature] = pd.factorize(data[feature], sort=True)[0]

# para hacer más OOS que el 1 año definido inicialmente
years_OOS = 5
YEAR = 52

train_period_length = 216
test_period_length = 12
# MultipleTimeSeriesCV siempre empieza por el final por eso tomará como periodo de validación/teste desde la ultima fecha que le pasemos hasta
# los años que definamos por n_splits
n_splits = int(YEAR * years_OOS / test_period_length)
lookahead = 1

cv = MultipleTimeSeriesCV(
    n_splits=n_splits,
    test_period_length=test_period_length,
    lookahead=lookahead,
    train_period_length=train_period_length,
)

for train_idx, test_idx in cv.split(X=data):
    pass
