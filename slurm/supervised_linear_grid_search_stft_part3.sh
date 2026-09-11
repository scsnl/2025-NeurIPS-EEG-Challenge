#!/bin/bash

#SBATCH --partition=owners
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --error=jobs/%x_%A_%a.err


# Check these!!!
# -----------------------------------------------------------------------------
# Array = 550
#SBATCH --array=0-549
#SBATCH --time=1-0
#SBATCH --mem=200G
#SBATCH --job-name=stft_part3

source "${EEGCHALLENGE_ENV}"
script="${EEGCHALLENGE_REPO}/experiments/challenge1/grid_search_stft_1101_part3.py"
# -----------------------------------------------------------------------------


set -eo pipefail
ml ruse
ruse -s --stdout python -u "$script"
