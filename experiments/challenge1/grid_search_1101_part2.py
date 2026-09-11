import sys
from pathlib import Path
from os import getenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

from eegchallenge.slurm import submit
from eegchallenge.features import reshape
from eegchallenge.pipeline import grid_search
from eegchallenge import config


# Set up output directories:
TO = config.grid_search_dir(challenge=1)

# Define grid to search:
PIPE = Pipeline(
    [
        ("first_scale", "passthrough"),
        ("frequency", "passthrough"),
        ("reshape", "passthrough"),
        ("second_scale", "passthrough"),
        ("model", "passthrough"),
    ],
    verbose=True,
)
COMMON = {
    "first_scale": ["passthrough"],
    "frequency": ["passthrough"],
    "reshape": [FunctionTransformer(reshape)],
    "second_scale": ["passthrough"],
}
PARAM_GRID = [
    {
        "model": [HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=100, random_state=42, max_iter=10000, scoring="neg_root_mean_squared_error")],
        "model__loss": ["absolute_error"],
        "model__learning_rate": [0.02, 0.05],
        "model__l2_regularization": [50, 200],
        "model__max_leaf_nodes": [127, 255],
        "model__min_samples_leaf": [800, 1600, 3200],
        "model__max_features": [0.2, 0.5, 1.0],
        "model__max_bins": [32],
        "model__max_depth": [3],
    },
    {
        "model": [HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=100, random_state=42, max_iter=10000, scoring="neg_root_mean_squared_error")],
        "model__loss": ["squared_error"],
        "model__learning_rate": [0.0025, 0.005, 0.01, 0.02, 0.05],
        "model__l2_regularization": [10, 50, 100, 200, 400],
        "model__max_leaf_nodes": [127, 255],
        "model__min_samples_leaf": [800, 1600],
        "model__max_features": [0.005],
        "model__max_bins": [63],
        "model__max_depth": [3, 6, None],
    },
    {
        "model": [HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=100, random_state=42, max_iter=10000, scoring="neg_root_mean_squared_error")],
        "model__loss": ["squared_error"],
        "model__learning_rate": [0.005, 0.0075, 0.01, 0.0125, 0.015],
        "model__l2_regularization": [0.1, 1, 10, 100],
        "model__max_leaf_nodes": [63, 127],
        "model__min_samples_leaf": [400, 800, 1200, 1600, 3200],
        "model__max_features": [0.01],
        "model__max_bins": [63],
        "model__max_depth": [None],
    },
    {
        "model": [HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=100, random_state=42, max_iter=20000, scoring="neg_root_mean_squared_error")],
        "model__loss": ["squared_error"],
        "model__learning_rate": [0.0015, 0.0025, 0.005, 0.01, 0.02],
        "model__l2_regularization": [0, 10, 100, 1000, 5000],
        "model__max_leaf_nodes": [31],
        "model__min_samples_leaf": [50],
        "model__max_features": [0.02, 0.2, 0.5, 1.0],
        "model__max_bins": [32],
        "model__max_depth": [1],
    },
]
PARAM_GRID = [i | COMMON for i in PARAM_GRID]


def main():
    grid_search(PIPE, PARAM_GRID, TO)


if __name__=='__main__':
    print('a:', f'0-{int(len(ParameterGrid(PARAM_GRID))) - 1}')
    print('J:', Path(__file__).stem)
    print('path:', __file__)
    SUBMIT_MODE = submit(
        a = f'0-{int(len(ParameterGrid(PARAM_GRID))) - 1}',
        t = '1-0',
        mem = '70G',
        J = Path(__file__).stem,
        path = __file__,
    )
    if not SUBMIT_MODE:
        main()
