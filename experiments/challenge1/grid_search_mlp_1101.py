import sys
from pathlib import Path
from os import getenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import TruncatedSVD

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
        ("reduce", "passthrough"),   # <— new step
        ("model", "passthrough"),
    ],
    verbose=True,
)
COMMON = {
    "first_scale": ["passthrough"],
    "frequency": ["passthrough"],
    "reshape": [FunctionTransformer(reshape)],
    "second_scale": ["passthrough"],
    "reduce": [TruncatedSVD(random_state=42)],  # <— new
}
PARAM_GRID = [
    {
        # Base estimator (constant)
        "model": [
            MLPRegressor(
                max_iter=200,
                early_stopping=False,
                n_iter_no_change=20,
                random_state=42,
            )
        ],

        # 5 × 4 × 5 × 1 × 1 = 100 points
        "reduce__n_components": [256, 512, 1024, 1536, 2048],      # 5
        "model__hidden_layer_sizes": [(8,), (16,), (24,), (32,)],   # 4
        "model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],             # 5
        "model__activation": ["relu"],                               # 1
        "model__learning_rate_init": [3e-4],                         # 1
    },
]

# Keep your existing merge pattern:
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
        part = 'owners',
    )
    if not SUBMIT_MODE:
        main()
