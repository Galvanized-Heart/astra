#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --output=slurm_logs/%j/cv_runner-%j.out
#SBATCH --error=slurm_logs/%j/cv_runner-%j.err
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=10:00:00
#SBATCH --mem=12G

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

export HYDRA_FULL_ERROR=1

# Change to the project directory to ensure all paths are correct
cd "/home/maxkirby/scratch/astra" || exit 1 # This path is specific to Fir!!

#echo "Running Python script with arguments:"
#echo "$@"

# Run training for k-fold cross validation
uv run src/astra/pipelines/hydra_train.py \
    architecture=cpi_pred_cross_attn \
    data=fold_1 \
    target_columns=km_only \
    recomposition=none \
    wandb.group=CpiPredCrossAttnModel-KM-none-valid/KM_Pearson-top0 \
    model.lightning_module.lr=0.00017385455229518664 \
    data.batch_size=64 \
    model.lightning_module.loss_weights.w_kcat_logit=-0.9290070049229996 \
    model.lightning_module.loss_weights.w_km_logit=1.2683505416498284 \
    model.lightning_module.loss_weights.w_ki_logit=0.6272212179795478 \
    architecture.params.n_heads=4 \
    architecture.params.d_ff=256