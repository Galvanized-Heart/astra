#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p slurm_logs

echo "Starting SLURM Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

# Set cache
export SCRATCH_CACHE_DIR="/gpfs/fs0/scratch/m/mahadeva/maxkirby/.cache"
export WANDB_DATA_DIR="$SCRATCH_CACHE_DIR/wandb-data"
export WANDB_CACHE_DIR="$SCRATCH_CACHE_DIR/wandb"
export WANDB_CONFIG_DIR="$SCRATCH_CACHE_DIR/wandb-config"
export WANDB_DIR="$SCRATCH_CACHE_DIR/wandb-logs"      
export HF_HOME="${SCRATCH_CACHE_DIR}/huggingface"
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$HF_HOME"

export HYDRA_FULL_ERROR=1

# Change to the project directory to ensure all paths are correct
cd "/gpfs/fs0/scratch/m/mahadeva/maxkirby/astra" || exit 1

echo "Running Python script with arguments:"
echo "$@"

# Run training for k-fold cross validation
uv run src/astra/pipelines/hydra_train.py "$@"