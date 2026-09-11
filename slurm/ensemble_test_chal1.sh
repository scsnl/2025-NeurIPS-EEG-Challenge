#!/bin/bash

# Usage: ./ensemble_test_chal1.sh [ensemble_method] [model1,model2,model3,...]
# Examples:
#   ./ensemble_test_chal1.sh mean "EEGNeX,BIOT,EEGConformer"
#   ./ensemble_test_chal1.sh inverse_loss "EEGNeX,BIOT,EEGConformer"
#   ./ensemble_test_chal1.sh weighted "EEGNeX,BIOT"
#   ./ensemble_test_chal1.sh  (uses defaults: mean method with EEGNeX,BIOT,EEGConformer)

ENSEMBLE_METHOD=${1:-"mean"}
ENSEMBLE_MODELS=${2:-"ATCNet,CTNet,Deep4Net,EEGNet,EEGNeX,EEGSimpleConv,Labram,MSVTNet,SPARCNet,SyncNet,BIOT,EEGConformer"}

echo "Submitting ensemble test job..."
echo "Ensemble method: $ENSEMBLE_METHOD"
echo "Models: $ENSEMBLE_MODELS"

# Convert comma-separated list to space-separated for command line
MODELS_SPACE=$(echo $ENSEMBLE_MODELS | tr ',' ' ')

# Create temporary SLURM script
cat > temp_ensemble_${ENSEMBLE_METHOD}.slurm << EOF
#!/bin/bash
#SBATCH --job-name=ensemble_${ENSEMBLE_METHOD}
#SBATCH --output=${EEGCHALLENGE_LOGS}/ensemble/ensemble_${ENSEMBLE_METHOD}_%j.out
#SBATCH --error=${EEGCHALLENGE_LOGS}/ensemble/ensemble_${ENSEMBLE_METHOD}_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,owners,normal,menon
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Change to scripts directory first
cd "${EEGCHALLENGE_REPO}"

source "${EEGCHALLENGE_ENV}"

# Create log directory if it doesn't exist
mkdir -p ${EEGCHALLENGE_LOGS}/ensemble

# Run ensemble test
python ${EEGCHALLENGE_REPO}/experiments/challenge1/ensembling_test.py \\
    --data_folder \${EEGCHALLENGE_DATA} \\
    --results_folder \${EEGCHALLENGE_RESULTS}/braindecode_all/challenge_1/2/ \\
    --ensemble_models ${MODELS_SPACE} \\
    --ensemble_method ${ENSEMBLE_METHOD} \\
    --batch_size 1024 \\
    --window_length 2.0 \\
    --shift 0.5 \\
    --valid_frac 0.1 \\
    --test_frac 0.1 \\
    --cache_dataset \\
    --load_cached_dataset \\
    --cache_dir \${EEGCHALLENGE_DATA}/braindecode_cache/challenge_1 \\
    --num_workers 4
EOF

# Submit the job
sbatch temp_ensemble_${ENSEMBLE_METHOD}.slurm
rm temp_ensemble_${ENSEMBLE_METHOD}.slurm

echo "Ensemble test job submitted successfully!"
echo "Check logs at: ${EEGCHALLENGE_LOGS}/ensemble/"
