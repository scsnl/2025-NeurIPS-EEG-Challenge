#!/bin/bash
# Environment setup. Copy to paths.sh, edit for your account, and source it
# before running anything:
#
#     cp config/paths.example.sh config/paths.sh
#     $EDITOR config/paths.sh
#     source config/paths.sh
#
# config/paths.sh is gitignored, so your local paths stay out of the repo.
# Batch jobs re-source this file (via EEGCHALLENGE_ENV) because they start from
# a clean shell.

# --- where this repository lives -------------------------------------------
export EEGCHALLENGE_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export EEGCHALLENGE_ENV="${EEGCHALLENGE_REPO}/config/paths.sh"

# --- data ------------------------------------------------------------------
# Raw HBN-EEG BIDS releases: the directory containing ds005505-bdf, ds005506-bdf,
# and so on. Read-only input; obtain it under the HBN data use agreement (see
# README "Data").
export EEGCHALLENGE_DATA="/path/to/HBN-EEG"

# --- working storage -------------------------------------------------------
# Large regenerable intermediates: the .npy arrays (hundreds of GB) and
# grid-search pickles. Must be scratch-class storage, NOT home or group home.
#
# On Sherlock, $SCRATCH is purged after 90 days without modification. That is
# what deleted the original intermediates; regenerate with pipelines/preprocess.py.
export EEGCHALLENGE_WORK="${SCRATCH}/2025_eeg_challenge"

# Slurm .out/.err files, and human-scale outputs worth keeping.
export EEGCHALLENGE_JOBS="${EEGCHALLENGE_WORK}/jobs"
export EEGCHALLENGE_LOGS="${EEGCHALLENGE_WORK}/logs"
export EEGCHALLENGE_RESULTS="${EEGCHALLENGE_WORK}/results"

mkdir -p "${EEGCHALLENGE_JOBS}" "${EEGCHALLENGE_LOGS}" "${EEGCHALLENGE_RESULTS}"

# --- optional: EEGPT backbone ----------------------------------------------
# Only needed for --model_name EEGPT. Neither the upstream code nor the
# checkpoint is redistributed here; see README "External model weights".
# export EEGPT_CHECKPOINT="/path/to/eegpt_mcae_58chs_4s_large4E.ckpt"
# export PYTHONPATH="/path/to/EEGPT/downstream:${PYTHONPATH}"

# --- python ----------------------------------------------------------------
# Replace with however you activate the environment (see environment.yml).
# On Sherlock the original runs used:
#   ml system libnvidia-container ruse
#   source /path/to/miniconda3/bin/activate /path/to/envs/2025_eeg_challenge
conda activate 2025_eeg_challenge
