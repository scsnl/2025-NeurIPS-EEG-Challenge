import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from numpy import std
from sklearn.metrics import root_mean_squared_error as rmse
from eegchallenge import config


# Set up output directories:
DATA_DIR = config.arrays_dir(2)
TO = config.grid_search_dir(challenge=2, job_id="lightGBM")


# def train():
#     train_data=lgb.Dataset(DATA_DIR/"train.csv")
#     valid=lgb.Dataset(DATA_DIR/"valid.csv", reference=train_data)
#     params = {
#         "objective":"regression","metric":"rmse",
#         "learning_rate":0.05,         # default 0.1 → slower, safer
#         "num_leaves":256,              # default 31 → more capacity
#         "max_depth":-1,                # default -1
#         "min_data_in_leaf":2000,       # default 20 → curb overfit on 700k rows
#         "feature_fraction":0.8,        # default 1.0 → less RAM/overfit
#         "bagging_fraction":0.8, "bagging_freq":1,  # default off → row subsample
#         "lambda_l2":1.0,               # default 0 → regularize
#         "max_bin":128,                 # default 255 → lower memory
#         "two_round":True,              # default False → lower peak build RAM
#         "histogram_pool_size":4096,    # default unlimited → cap (~4 GB)
#         "seed":0
#     }
#     bst = lgb.train(params, train_data, num_boost_round=2000,
#                 valid_sets=[valid],
#                 callbacks=[lgb.early_stopping(stopping_rounds=100)])
#     bst.save_model(TO/"lgb.txt")


def train():
    
    y_train = np.load(DATA_DIR/'y_train.npy', mmap_mode='r')
    y_val = np.load(DATA_DIR/'y_val.npy', mmap_mode='r')
    X_train = np.load(DATA_DIR/'X_train.npy', mmap_mode='r')
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_val = np.load(DATA_DIR/'X_val.npy', mmap_mode='r')
    X_val = X_val.reshape(X_val.shape[0],-1)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective":"regression","metric":"rmse",
        "learning_rate":0.05,         # default 0.1 → slower, safer
        "num_leaves":256,              # default 31 → more capacity
        "max_depth":-1,                # default -1
        "min_data_in_leaf":2000,       # default 20 → curb overfit on 700k rows
        "feature_fraction":0.8,        # default 1.0 → less RAM/overfit
        "bagging_fraction":0.8, "bagging_freq":1,  # default off → row subsample
        "lambda_l2":1.0,               # default 0 → regularize
        "max_bin":128,                 # default 255 → lower memory
        "two_round":True,              # default False → lower peak build RAM
        "histogram_pool_size":4096,    # default unlimited → cap (~4 GB)
        "seed":0
    }

    bst = lgb.train(params, train_data, num_boost_round=2000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=100)])
    bst.save_model(TO/"lgb_1.txt")


def score():

    bst = lgb.Booster(model_file=str(TO / "lgb_1.txt"))
    num_it = getattr(bst, "best_iteration", None) or bst.num_trees()
    
    y_val = np.load(DATA_DIR/'y_val.npy', mmap_mode='r')
    X_val = np.load(DATA_DIR/'X_val.npy', mmap_mode='r')
    X_val = X_val.reshape(X_val.shape[0],-1)
    y_test = np.load(DATA_DIR/'y_test.npy', mmap_mode='r')
    X_test = np.load(DATA_DIR/'X_test.npy', mmap_mode='r')
    X_test = X_test.reshape(X_test.shape[0],-1)

    for X,y in [(X_val,y_val), (X_test,y_test)]:
        y_pred = bst.predict(X, num_iteration=num_it)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        nrmse = rmse(y, y_pred) / std(y)
        print("r2:", r2)
        print('nrmse:', nrmse)
        print('r2 from nrmse:', 1-(nrmse**2))
        print('r:', np.corrcoef(y_pred,y))


def main():
    train()
    score()


if __name__=='__main__':
    main()
