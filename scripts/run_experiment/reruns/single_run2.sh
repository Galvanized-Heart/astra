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

uv run src/astra/pipelines/hydra_train.py experiment_mode=multi_task/direct architecture=linear data=fold_4 model.lightning_module.mtl_strategy=uncertainty wandb.group=Uncertainty-Direct-Linear extra_tags=[uncertainty_exp,5fcv] trainer.epochs=20 model.lightning_module.lr=0.0013374409661630064 data.batch_size=64 architecture.params.dim_1=512 architecture.params.dim_2=128