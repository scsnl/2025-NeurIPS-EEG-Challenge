#!/bin/bash

MODEL_SELECTION=${1:-"EEGNeX, BIOT"}

# Function to get model parameters from JSON
get_model_params() {
    local model_name=$1
    jq -r --arg model "$model_name" '.models[] | select(.model_name == $model) | "\(.learning_rate) \(.weight_decay) \(.num_epochs) \(.batch_size) \(.window_length) \(.shift)"' "${EEGCHALLENGE_REPO}/config/model_config.json"
}

# Function to submit job for a single model
submit_model() {
    local model_name=$1
    local params=$(get_model_params $model_name)
    
    if [ -z "$params" ]; then
        echo "Error: Model '$model_name' not found in config"
        return 1
    fi
    
    read -r lr wd epochs batch_size window_length shift <<< "$params"
    
    # Create temporary SLURM script
    cat > temp_${model_name}.slurm << EOF
#!/bin/bash
#SBATCH --job-name=mae_${model_name}
#SBATCH --output=${EEGCHALLENGE_LOGS}/chal_1/all/mae/mae_${model_name}_%a.out
#SBATCH --error=${EEGCHALLENGE_LOGS}/chal_1/all/mae/mae_${model_name}_%a.err
#SBATCH --array=2
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,owners
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

source "${EEGCHALLENGE_ENV}"
ml load cuda/11.7.1

# Set seed based on array task ID
SEED=\$((1234 * SLURM_ARRAY_TASK_ID))

# Run MAE pretraining with model-specific parameters
python "${EEGCHALLENGE_REPO}/pipelines/pretrain_mae_challenge1.py" \\
    --data_folder \${EEGCHALLENGE_DATA} \\
    --results_folder \${EEGCHALLENGE_RESULTS}/braindecode_all/challenge_1/mae/\${SLURM_ARRAY_TASK_ID} \\
    --model_name ${model_name} \\
    --task_name ThePresent \\
    --learning_rate 1e-3 \\
    --weight_decay 1e-4 \\
    --num_epochs 50 \\
    --batch_size 1024 \\
    --window_length ${window_length} \\
    --shift ${shift} \\
    --early_stopping_patience 20 \\
    --min_delta 1e-4 \\
    --valid_frac 0.1 \\
    --test_frac 0.1 \\
    --seed \${SEED} \\
    --use_all_releases \\
    --noise_std 0.1 \\
    --enable_augmentation \\
    --cache_dataset \\
    --load_cached_dataset \\
    --cache_dir \${EEGCHALLENGE_DATA}/braindecode_cache/challenge_1 \\
    --save_metrics \\
    --save_plots \\
    --save_model \\
    --num_workers 4 \\
    --iter \${SLURM_ARRAY_TASK_ID} \\
    --decoder transformer \\
    --patch_size 25 \\
    --mask_ratio 0.50 \\
    --mask_seed \${SEED} \\
    --normalize_data
EOF

    # Submit the job
    sbatch --gpu_cmode=shared temp_${model_name}.slurm
    rm temp_${model_name}.slurm
}

# Get model list
if [ "$MODEL_SELECTION" = "all" ]; then
    MODELS=$(jq -r '.models[].model_name' "${EEGCHALLENGE_REPO}/config/model_config.json" | tr '\n' ' ')
else
    MODELS=$(echo $MODEL_SELECTION | tr ',' ' ')
fi

echo "Submitting MAE pretraining jobs for models: $MODELS"

# Submit jobs for each model
for model in $MODELS; do
    echo "Submitting MAE pretraining job for $model..."
    submit_model $model
done

echo "All MAE pretraining jobs submitted successfully!"
