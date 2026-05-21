#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --output=slurm_logs/%j/cv_runner-%j.out
#SBATCH --error=slurm_logs/%j/cv_runner-%j.err
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=15:00:00
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

# Change to the project directory to ensure all paths are correct
cd "/home/maxkirby/scratch/astra" || exit 1 # This path is specific to Fir!!

uv run src/astra/pipelines/hydra_train.py architecture=linear data=fold_4 target_columns=all recomposition=advanced wandb.group=LinearBaselineModel-all-advanced-valid/kcat_Pearson-top0 model.lightning_module.lr=0.0002370154508334259 data.batch_size=64 model.lightning_module.loss_weights.w_kcat_logit=0.8631742499833921 model.lightning_module.loss_weights.w_km_logit=-0.3895040747337566 model.lightning_module.loss_weights.w_ki_logit=-0.617979972411193 architecture.params.dim_1=512 architecture.params.dim_2=64
