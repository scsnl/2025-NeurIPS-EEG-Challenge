"""Grid-search driver for the scikit-learn track.

Each script in ``experiments/`` defines a ``Pipeline`` and a parameter grid,
then calls :func:`grid_search` here. The work is split across a Slurm job array:
:func:`~eegchallenge.slurm.get_job` hands each array member its slice of the
grid, and each member writes ``{SLURM_ARRAY_TASK_ID}.pkl`` into a shared
per-job directory.

Model selection uses :class:`~sklearn.model_selection.PredefinedSplit` rather
than k-fold, because the train/validation boundary is a fixed subject-level
split produced by ``pipelines/preprocess.py`` — resampling it would leak across
subjects.

.. note:: **Split definition changed after the competition.**

   The arrays are accompanied by ``test_fold.npy``, which marks each row as
   ``-1`` (train) or ``0`` (select on). This module expects the current
   definition::

       train      = X_train
       select on  = X_val

   The November 2025 competition submissions were produced with::

       train      = X_train + X_val
       select on  = X_test   (release R5)

   That is, hyperparameters were originally selected against R5. This was
   harmless for the leaderboard, which scored against a hidden R12 set, but it
   means any figure computed on R5 with the original code is optimistic. The
   code as submitted is preserved verbatim under ``submission/as_submitted/``.
   See the README section "Reproducibility".
"""

from os import getenv

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit

from . import config
from .slurm import get_job

__all__ = ["grid_search", "grid_search_raise", "fit_score", "load_X_y_test_fold"]


def load_X_y_test_fold(challenge=1):
    """Memory-map the combined design matrix and its predefined split.

    Returns ``(X, y, test_fold)`` where ``test_fold`` is ``-1`` for training
    rows and ``0`` for the selection rows, as consumed by ``PredefinedSplit``.

    The arrays are memory-mapped: ``X`` for challenge 1 is roughly 100k x 129 x
    200 float32, far too large to hold in RAM alongside a fitted model.
    """
    path = config.arrays_dir(challenge)
    X = np.load(path / "X.npy", mmap_mode="r")
    y = np.load(path / "y.npy", mmap_mode="r")
    test_fold = np.load(path / "test_fold.npy", mmap_mode="r")
    # Guard against a stale or partially-rewritten combined array.
    assert (test_fold == -1).sum() == np.load(path / "X_train.npy", mmap_mode="r").shape[0]
    assert (test_fold == 0).sum() == np.load(path / "X_val.npy", mmap_mode="r").shape[0]
    return X, y, test_fold


def _run(pipe, param_grid, to, challenge, error_score):
    # Take only this array member's slice of the grid.
    param_grid = ParameterGrid(param_grid)
    param_grid = [param_grid[i] for i in get_job(len(param_grid))]
    param_grid = [{k: [v] for k, v in d.items()} for d in param_grid]  # GridSearchCV wants lists

    X, y, test_fold = load_X_y_test_fold(challenge=challenge)
    cv = PredefinedSplit(test_fold=test_fold)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        refit=True,
        n_jobs=1,  # n_jobs>1 duplicates the memory-mapped X per worker and blows up memory
        scoring="r2",
        verbose=4,
        return_train_score=True,
        **({"error_score": error_score} if error_score is not None else {}),
    )
    grid.fit(X, y)

    if to is None:
        to = config.grid_search_dir(challenge)
    joblib.dump(grid, to / f"{getenv('SLURM_ARRAY_TASK_ID')}.pkl")
    return grid


def grid_search(pipe, param_grid, to=None, challenge=1):
    """Fit this array member's slice of ``param_grid``; failures score as NaN."""
    return _run(pipe, param_grid, to, challenge, error_score=None)


def grid_search_raise(pipe, param_grid, to=None, challenge=1):
    """As :func:`grid_search`, but a failing fit raises instead of scoring NaN.

    Used when debugging a new estimator, where a silent NaN would look like a
    merely bad hyperparameter rather than a broken pipeline.
    """
    return _run(pipe, param_grid, to, challenge, error_score="raise")


def fit_score(pipe, to, challenge=1):
    """Fit one fixed pipeline on the training rows and report the selection score."""
    X, y, test_fold = load_X_y_test_fold(challenge=challenge)
    train_mask = test_fold == -1
    val_mask = test_fold == 0
    pipe.fit(X[train_mask], y[train_mask])
    print(pipe.score(X[val_mask], y[val_mask]))
    joblib.dump(pipe, to)
    return pipe
