"""Turn raw HBN-EEG BIDS releases into the .npy arrays the sklearn track uses.

Original: scripts/challenges/challenge2/supervised_linear_preprocess.py
Unchanged except that hardcoded paths now come from eegchallenge.config.

Run via slurm/preprocess.sh -- this needs bigmem (see the note on preprocess()).
"""

import sys
import gc
from os import getenv
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import signal
from scipy.fft import rfft,rfftfreq

from matplotlib import pyplot as plt

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import joblib
from sklearn.linear_model import SGDRegressor,RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.model_selection import GridSearchCV,ParameterGrid,PredefinedSplit

from tqdm import tqdm

from eegchallenge.parser import config_call
from eegchallenge.data import load_and_process_data
from eegchallenge.io import create_directories, print_config
from eegchallenge.slurm import get_job
from eegchallenge.config import data_root, arrays_dir


FS = 100
NPERSEG = 100


def save_train_val_test_pt(challenge=2):
    assert challenge in [1,2]
    out = arrays_dir(challenge, create=True)
    config = dict(
        data_folder=str(data_root()),
        use_all_releases=True, # Makes release 5 the test and everything else the train
        data_releases=["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"],
        eval_release="R5",
        task_name='contrastChangeDetection',
        window_length=2.0, # default in load_and_process_data
        shift=0.5, # default in load_and_process_data
        valid_frac=0.2, # use_all_releases=True means valid_frac=0.2
        # test_frac=0.1, # Not relevant when use_all_releases=True
        # results_folder:
        # model_name:
    )
    augmentation_params = None
    train_set, val_set, test_set = config_call(load_and_process_data, config, challenge=challenge, augmentation_params=augmentation_params)
    print('train save start')
    torch.save(train_set, out/'train_set.pt')
    print('train save success')
    torch.save(val_set, out/'val_set.pt')
    print('val save success')
    torch.save(test_set, out/'test_set.pt')
    print('test save success')


def save_X_y(challenge=2):
    assert challenge in [1,2]
    out = arrays_dir(challenge, create=True)
    for name in ['train','val','test']:
        data_set = torch.load(out/f'{name}_set.pt', weights_only=False)
        loader = DataLoader(data_set, batch_size=len(data_set))
        if challenge==1:
            X, y, _ = next(iter(loader))
        else:
            X, y = next(iter(loader))
        np.save(out/f'X_{name}.npy', X)
        np.save(out/f'y_{name}.npy', y)


def load_data(split, challenge=1):
    assert split in ['train','val','test']
    assert challenge in [1,2]
    path = arrays_dir(challenge)
    X = np.load(path/f'X_{split}.npy')
    y = np.load(path/f'y_{split}.npy')
    y = y.squeeze()
    # if split=='train':
    #     X = np.concatenate((X, np.load(path/f'X_val.npy')), axis=0)
    #     y = np.concatenate((y, np.load(path/f'y_val.npy')), axis=0)
    if challenge==1:
        if split=='train':
            batch_dim = 80889
        elif split=='val':
            batch_dim = 19883
        elif split=='test':
            batch_dim = 15144
        assert X.shape==(batch_dim,129,200)
        assert y.shape==(batch_dim,)
    return X, y


def save_X_y_combined(challenge=1):
    path = arrays_dir(challenge, create=True)
    X_train, y_train = load_data('train', challenge=challenge)
    X_test, y_test = load_data('val', challenge=challenge)
    assert X_train.shape[0]==y_train.shape[0]
    assert X_test.shape[0]==y_test.shape[0]
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    test_fold = [-1]*X_train.shape[0] + [0]*X_test.shape[0]
    np.save(path/'X.npy', X)
    np.save(path/'y.npy', y)
    np.save(path/'test_fold.npy', test_fold)


def preprocess(challenge, final_only=False):
    """Takes ~60 min and >40 GB RAM."""
    if not final_only:
        print('save_train_val_test_pt starts!')
        save_train_val_test_pt(challenge=challenge)
        print('save_X_y starts!')
        save_X_y(challenge=challenge)
    print('save_X_y_combined starts!')
    save_X_y_combined(challenge=challenge)


def test(challenge):
    assert challenge==1
    print('test starts!')
    path = arrays_dir(challenge)
    X = np.load(path/'X.npy')
    y = np.load(path/'y.npy')
    test_fold = np.load(path/'test_fold.npy')
    assert X.shape[0]==y.shape[0]
    assert X.shape[0]==(80889+19883)
    assert (test_fold==-1).sum()==80889
    assert (test_fold==0).sum()==19883


def main():
    preprocess(challenge=2, final_only=False)
    # test(challenge=1)


if __name__=='__main__':
    main()
