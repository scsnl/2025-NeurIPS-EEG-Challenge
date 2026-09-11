import os
import glob
import re
from typing import List

from .slurm import jobs_dir


MEMORY_PATTERN = re.compile(r'(?mi)^\s*Memory:\s*([0-9]+(?:\.[0-9]+)?)\s*GB\b')


def collect_memory_gb_values(pattern: str) -> List[float]:
    """
    Scan all files matching pattern and return a list of
    memory usage values in GB (floats). If a file has multiple 'Memory:' lines,
    the last value is used. Files without a match are skipped.
    """
    pattern = os.path.join(str(jobs_dir(create=False)), pattern)

    values: List[float] = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        matches = MEMORY_PATTERN.findall(text)
        if matches:
            values.append(float(matches[-1]))

    return values
