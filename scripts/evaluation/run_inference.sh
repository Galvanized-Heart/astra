#!/bin/bash
#SBATCH --job-name=astra_inference
#SBATCH --output=slurm_logs/%j/inference-%j.out
#SBATCH --error=slurm_logs/%j/inference-%j.err
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G

# Create a logs directory if it doesn't exist
mkdir -p slurm_logs

echo "Starting SLURM Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

# Set cache
export SCRATCH_CACHE_DIR=".cache"
export WANDB_DATA_DIR="$SCRATCH_CACHE_DIR/wandb-data"
export WANDB_CACHE_DIR="$SCRATCH_CACHE_DIR/wandb"
export WANDB_CONFIG_DIR="$SCRATCH_CACHE_DIR/wandb-config"
export WANDB_DIR="$SCRATCH_CACHE_DIR/wandb-logs"      
export HF_HOME="${SCRATCH_CACHE_DIR}/huggingface"
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$HF_HOME"

# Set HF env variables
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

# Ensure API key is available for wandb downloads
# (Uncomment and add your key if it isn't saved in your ~/.netrc on the compute nodes)
# export WANDB_API_KEY="your_40_char_api_key"

# Change to the project directory
cd "/home/maxkirby/scratch/astra" || exit 1

echo "Running Inference Pipeline..."

# Run the inference script
uv run src/astra/pipelines/evaluation/run_inference.py \
    --manifest "results/inference_manifest.json" \
    --out_dir "results/inference_data"

echo "Job finished at: $(date)"