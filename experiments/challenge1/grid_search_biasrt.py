import sys
from pathlib import Path
from os import getenv
import glob
from joblib import load

import numpy as np

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.utils import get_tags

from eegchallenge.sklearn_models import BiasRTRegressor, closest, towards_mean, away_mean
from eegchallenge.features import power_phase, reshape, mad_scaling_clipping_channel_trial, power_phase_temporal, functional_connectivity, mad_scaling_clipping_trial_time, real_cepstrum, complex_cepstrum, enveloppe
from eegchallenge.pipeline import grid_search, grid_search_raise
from eegchallenge import config

def load_estimators(paths):
    return [(Path(path).stem, load(path)) for path in paths]
ensemble_paths = sorted(glob.glob(str(config.grid_search_dir(challenge=1, job_id="8865913_8867010__top10", create=False) / "*.pkl")), key=lambda s: (len(s), s))

estimators = load_estimators(ensemble_paths)

# Set up output directories:
TO = config.grid_search_dir(challenge=1)
( TO / 'cache' / str(getenv('SLURM_ARRAY_TASK_ID')) ).mkdir(parents=True, exist_ok=True)

# Define grid to search:
PIPE = Pipeline(
    [
        ("model", "passthrough"),
    ],
    verbose=True,
)
PARAM_GRID = [
    {
        "model": [BiasRTRegressor(estimator = estimators[0][1], threshold = 15, strategy = away_mean)],
        "model__threshold": [15],
    },
]

def main():
    #grid_search(PIPE, PARAM_GRID, TO)
    grid_search_raise(PIPE, PARAM_GRID, TO)

if __name__=='__main__':
    main()