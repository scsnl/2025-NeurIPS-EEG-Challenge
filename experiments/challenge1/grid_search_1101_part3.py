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
        "model": [HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=100, random_state=42, max_iter=20000, scoring="neg_root_mean_squared_error")],
        "model__loss": ["squared_error"],
        "model__learning_rate": [0.0025, 0.005, 0.01, 0.02, 0.05],
        "model__l2_regularization": [0.1, 1, 10, 100, 400],
        "model__max_leaf_nodes": [127, 255],
        "model__min_samples_leaf": [100, 400, 1600, 3200],
        "model__max_features": [0.01, 0.1, 1.0],
        "model__max_bins": [32],
        "model__max_depth": [2],
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
        part='owners',
    )
    if not SUBMIT_MODE:
        main()
