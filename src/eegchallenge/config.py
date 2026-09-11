"""Filesystem locations, resolved from environment variables.

The original code hard-coded absolute paths under a single user's ``$OAK`` and
``$SCRATCH``. Everything now goes through this module so the pipeline runs for
any user on any cluster. Set the variables in ``config/paths.example.sh`` (copy
it to ``paths.sh`` and edit) or export them yourself.

    EEGCHALLENGE_DATA     raw HBN-EEG BIDS releases (read-only input)
    EEGCHALLENGE_WORK     large intermediates: .npy arrays, grid-search pickles
    EEGCHALLENGE_RESULTS  small outputs worth keeping: metrics, summaries

``EEGCHALLENGE_WORK`` should point at scratch-class storage. The arrays are
hundreds of GB and are regenerable, so they do not belong on backed-up space.
Note that Sherlock's ``$SCRATCH`` deletes files untouched for 90 days.
"""

import os
from pathlib import Path

__all__ = [
    "data_root",
    "work_root",
    "results_root",
    "arrays_dir",
    "grid_search_dir",
]


def _env_path(name, default=None):
    value = os.getenv(name)
    if value:
        return Path(value)
    if default is not None:
        return Path(default)
    raise RuntimeError(
        f"{name} is not set. Copy config/paths.example.sh to config/paths.sh, "
        f"edit it for your account, and `source` it before running."
    )


def data_root():
    """Raw HBN-EEG BIDS releases (the ``ds0055xx-bdf`` directories)."""
    return _env_path("EEGCHALLENGE_DATA")


def work_root():
    """Scratch space for large regenerable intermediates."""
    scratch = os.getenv("SCRATCH")
    default = f"{scratch}/2025_eeg_challenge" if scratch else None
    return _env_path("EEGCHALLENGE_WORK", default)


def results_root():
    """Small outputs worth keeping (metrics, summaries, checkpointed models)."""
    return _env_path("EEGCHALLENGE_RESULTS", work_root() / "results")


def arrays_dir(challenge, create=False):
    """Where ``X.npy`` / ``y.npy`` / ``test_fold.npy`` live for a challenge."""
    if challenge not in (1, 2):
        raise ValueError(f"challenge must be 1 or 2, got {challenge!r}")
    path = work_root() / f"challenge{challenge}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def grid_search_dir(challenge, job_id=None, create=True):
    """Output directory for one grid-search job.

    Results are grouped by Slurm array job ID so that concurrent array members
    write into the same directory without colliding (each writes
    ``{SLURM_ARRAY_TASK_ID}.pkl``). Falls back to ``local`` when not running
    under Slurm.
    """
    if job_id is None:
        job_id = os.getenv("SLURM_ARRAY_JOB_ID") or "local"
    path = arrays_dir(challenge) / "supervised_linear_grid_search" / str(job_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
