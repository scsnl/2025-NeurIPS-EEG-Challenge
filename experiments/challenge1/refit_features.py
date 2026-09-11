import sys
from pathlib import Path
from os import getenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

from eegchallenge.slurm import submit
from eegchallenge.features import reshape, power_phase_circular, enveloppe
from eegchallenge.pipeline import fit_score
from eegchallenge import config


# Set up output directories:
TO = config.grid_search_dir(challenge=1, job_id="refit_features")

# Define grid to search:
PIPES = [
    Pipeline(
        [
            ('first_scale', 'passthrough'),
            ('frequency', FunctionTransformer(power_phase_circular)),
            ('reshape', FunctionTransformer(reshape)),
            ('second_scale', 'passthrough'),
            ('model', HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=15, random_state=42, max_iter=300, l2_regularization=0, learning_rate=0.05, max_features=0.2, max_leaf_nodes=31, min_samples_leaf=20,))
        ],
        verbose=True
    ),
    Pipeline(
        [
            ('first_scale', 'passthrough'),
            ('frequency', FunctionTransformer(enveloppe)),
            ('reshape', FunctionTransformer(reshape)),
            ('second_scale', 'passthrough'),
            ('model', HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=15, random_state=42, max_iter=300, l2_regularization=0, learning_rate=0.05, max_features=0.2, max_leaf_nodes=63, min_samples_leaf=20,))
        ],
        verbose=True
    )
]


def main():
    fit_score(PIPES[0], TO/'ppcircular.pkl')
    fit_score(PIPES[1], TO/'enveloppe.pkl')


if __name__=='__main__':
    print('a:', '0')
    print('J:', Path(__file__).stem)
    print('path:', __file__)
    SUBMIT_MODE = submit(
        a = '0',
        t = '0-6',
        mem = '256G',
        J = Path(__file__).stem,
        path = __file__,
        part='bigmem',
    )
    if not SUBMIT_MODE:
        main()
