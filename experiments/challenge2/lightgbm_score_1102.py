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


def score():

    bst = lgb.Booster(model_file=str(TO / "lgb.txt"))
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
    score()


if __name__=='__main__':
    main()
