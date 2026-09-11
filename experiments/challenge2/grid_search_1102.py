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
TO = config.grid_search_dir(challenge=2)

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
        "model": [HistGradientBoostingRegressor(early_stopping=True,n_iter_no_change=15,scoring=None,random_state=42,max_iter=300)],
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [31, 63],
        "model__min_samples_leaf": [20],
        "model__l2_regularization": [0.0, 1e-3],
        "model__max_features": [0.05, 0.1, 0.2],
    },
]
PARAM_GRID = [i | COMMON for i in PARAM_GRID]


def main():
    grid_search(PIPE, PARAM_GRID, TO, challenge=2)


if __name__=='__main__':
    # print('a:', f'0-{int(len(ParameterGrid(PARAM_GRID))) - 1}')
    # print('J:', Path(__file__).stem)
    # print('path:', __file__)
    # SUBMIT_MODE = submit(
    #     # a = f'0-{int(len(ParameterGrid(PARAM_GRID))) - 1}',
    #     a = '0',
    #     # t = '1-0',
    #     t = '0-4',
    #     mem = '256G',
    #     J = Path(__file__).stem,
    #     path = __file__,
    #     # part = 'owners',
    #     part = 'bigmem',
    # )
    # if not SUBMIT_MODE:
    #     main()
    main()
