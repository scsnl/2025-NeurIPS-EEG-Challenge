#!/bin/bash

#SBATCH --partition=normal,owners
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --error=jobs/%x_%A_%a.err


# Check these!!!
# -----------------------------------------------------------------------------
# Array = 360
#SBATCH --array=0-359
#SBATCH --time=2-0
#SBATCH --mem=160G
#SBATCH --job-name=grid_search_ensemble

source "${EEGCHALLENGE_ENV}"
script="${EEGCHALLENGE_REPO}/experiments/challenge1/grid_search_ensembling.py"
# -----------------------------------------------------------------------------


set -eo pipefail
ml ruse
ruse -s --stdout python -u "$script"
