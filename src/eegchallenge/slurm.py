"""Slurm helpers for self-submitting scripts.

The grid-search scripts in ``experiments/`` submit *themselves*: running one
directly from a login node sbatches an array job sized to the parameter grid,
and each array member re-runs the same file with ``SLURM_ARRAY_TASK_ID`` set,
which is what makes :func:`submit` return ``False`` and the work actually run.

    python experiments/challenge1/grid_search_1101_part1.py   # submits ~2000 jobs

That is a lot of jobs from one innocuous-looking command. Check the printed
array size before running.
"""

import subprocess
from os import getenv
from pathlib import Path

from . import config

__all__ = ["submit", "get_job", "jobs_dir"]


def jobs_dir(create=True):
    """Where Slurm ``.out``/``.err`` files land."""
    path = Path(getenv("EEGCHALLENGE_JOBS") or config.work_root() / "jobs")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _setup_script():
    """Environment setup sourced inside the batch job.

    Required, because the batch job starts from a clean shell and needs the
    conda environment and ``EEGCHALLENGE_*`` variables before it can import
    this package.
    """
    value = getenv("EEGCHALLENGE_ENV")
    if not value:
        raise RuntimeError(
            "EEGCHALLENGE_ENV is not set. It must point at the shell script "
            "that activates the environment (see config/paths.example.sh); "
            "batch jobs source it before running Python."
        )
    return value


def submit(*, a, t, mem, J, path, part="normal,owners", o=r"%x_%A_%a.out", e=r"%x_%A_%a.err"):
    """Submit ``path`` as a Slurm array job, unless already running as one.

    Args:
        a: array spec, e.g. ``'0-199'``.
        t: walltime, ``D-HH`` or ``HH:MM:SS``.
        mem: memory per task, e.g. ``'70G'``.
        J: job name.
        path: script to run.
        part: partition list.

    Returns:
        ``True`` if a job was submitted (so the caller should exit), ``False``
        if we are inside an array member and the caller should do the work.
    """
    if getenv("SLURM_ARRAY_TASK_ID") is not None:
        return False

    out = jobs_dir()
    command = (
        f"source {_setup_script()};"
        f"sbatch --array={a} --time={t} --mem={mem} --job-name={J} "
        f"--partition={part} --ntasks=1 --cpus-per-task=1 "
        f"--output={out}/{o} --error={out}/{e} "
        f"--mail-type=FAIL "
        f'--wrap="ml ruse; ruse -s --stdout python -u {path}"'
    )
    subprocess.run(command, shell=True, check=True)
    print("Ran:", command)
    return True


def get_job(total_parallel_units):
    """Allocate unit indices to this job array member.

    Args:
        total_parallel_units: e.g. the number of grid points to cover.

    Returns:
        The indices this member is responsible for. Outside Slurm, all of them.
    """
    if getenv("SLURM_ARRAY_TASK_MIN") is not None:
        assert int(getenv("SLURM_ARRAY_TASK_MIN")) == 0
        cpu_index = int(getenv("SLURM_ARRAY_TASK_ID"))
        total_cpus = int(getenv("SLURM_ARRAY_TASK_MAX")) + 1
        return list(range(cpu_index, total_parallel_units, total_cpus))
    return list(range(total_parallel_units))
