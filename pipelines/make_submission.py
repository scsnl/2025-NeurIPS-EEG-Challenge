"""Build the constant-predictor baseline submission bundle.

Original: scripts/submission/pickle_models.py
Unchanged except that hardcoded paths now come from eegchallenge.config.

This emits the 'super_dumb' sanity submission -- a pipeline that ignores its
input and predicts a constant -- used to check the bundle format and the
evaluation harness agree before submitting a real model. See submission/README.md
for the bundle layout and for building a bundle around real fitted pipelines.
"""

from functools import partial
from datetime import date
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib
from subprocess import run

from eegchallenge import config

# Check !!!
# -----------------------------------------------------------------------------
SUBMISSION_NAME = 'super_dumb'
# -----------------------------------------------------------------------------
SUBMISSION_NAME = f"{date.today().strftime('%Y-%m-%d')}_{SUBMISSION_NAME}"
TO = config.results_root() / 'submission' / SUBMISSION_NAME
TO.mkdir(exist_ok=True, parents=True)


def challenge_1():
    lr = LinearRegression()
    lr.coef_ = np.array(np.zeros(129*200))
    lr.intercept_ = 1.6
    lr.n_features_in_ = lr.coef_.shape[0]
    return Pipeline(
        [
            ('reshape',FunctionTransformer(partial(np.reshape,shape=(-1,129*200)))),
            ('model',lr)
        ]
    )


def challenge_2():
    lr = LinearRegression()
    lr.coef_ = np.array(np.zeros(129*200))
    lr.intercept_ = 0.0 # 0.012334863140807591
    lr.n_features_in_ = lr.coef_.shape[0]
    return Pipeline(
        [
            ('reshape',FunctionTransformer(partial(np.reshape,shape=(-1,129*200)))),
            ('model',lr)
        ]
    )


def main():
    for model,name in [(challenge_1(),'challenge_1'), (challenge_2(),'challenge_2')]:
        joblib.dump(model, TO/f"{name}.pkl")
    cp_from = Path(__file__).resolve().parent.parent / 'submission'
    run(f"cp {cp_from/'submission_sklearn.py'} {TO/'submission.py'}", shell=True, check=True)
    run(f"cd {TO.parent}; zip -r {SUBMISSION_NAME}.zip {SUBMISSION_NAME}", shell=True, check=True)


if __name__=='__main__':
    main()
