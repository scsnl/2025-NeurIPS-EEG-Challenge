#!/bin/bash
set -eo pipefail

extract_jobid() {
  awk '/Submitted batch job/ {print $NF}'
}

scripts="${EEGCHALLENGE_REPO}"
out='${EEGCHALLENGE_JOBS}/preprocess_challenge_1_%j.out'
err='${EEGCHALLENGE_JOBS}/preprocess_challenge_1_%j.err'

source "$scripts/environment.sh"

# Job should succeed with 1500G, but if it fails, run with successively higher:
sbatch_out=$(sbatch --mem=256G --job-name=preprocess_challenge_1 --partition=bigmem --time=0-8 --ntasks=1 --cpus-per-task=1 --output=$out --error=$err --wrap="ml ruse; ruse -s --stdout python $scripts/challenges/challenge1/supervised_linear_preprocess.py" | extract_jobid)
sbatch_out=$(sbatch --mem=1500G --job-name=preprocess_challenge_1 --partition=bigmem --time=0-8 --ntasks=1 --cpus-per-task=1 --output=$out --error=$err --wrap="ml ruse; ruse -s --stdout python $scripts/challenges/challenge1/supervised_linear_preprocess.py" --dependency=afternotok:$sbatch_out | extract_jobid)
