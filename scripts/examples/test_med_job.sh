#!/bin/bash
#SBATCH --job-name="hpo-job"
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=5:00:00
#SBATCH --mem=64G

# Getting training speed test on Fir's H100 for one of Max's HPO jobs

# Set cache
export SCRATCH_CACHE_DIR="home/maxkirby/scratch/.cache"
export WANDB_DATA_DIR="$SCRATCH_CACHE_DIR/wandb-data"
export WANDB_CACHE_DIR="$SCRATCH_CACHE_DIR/wandb"
export WANDB_CONFIG_DIR="$SCRATCH_CACHE_DIR/wandb-config"
export WANDB_DIR="$SCRATCH_CACHE_DIR/wandb-logs"      
export HF_HOME="${SCRATCH_CACHE_DIR}/huggingface"
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$HF_HOME"

export HYDRA_FULL_ERROR=1

uv run /scratch/maxkirby/astra/src/astra/pipelines/hydra_train.py \
    architecture=cpi_pred_conv \
    data=fold_2 \
    target_columns=all \
    recomposition=basic \
    wandb.group=CpiPredConvModel-all-basic-valid/kcat_Pearson-top0 \
    model.lightning_module.lr=0.00011850028139507138 \
    data.batch_size=64 \
    model.lightning_module.loss_weights.w_kcat_logit=1.0690237342069868 \
    model.lightning_module.loss_weights.w_km_logit=-0.056844873968354026 \
    model.lightning_module.loss_weights.w_ki_logit=1.0935193892314623 \
    architecture.params.hid_dim=64 \
    architecture.params.kernal_1=5 \
    architecture.params.conv_out_dim=64 \
    architecture.params.kernal_2=3 \
    architecture.params.last_hid=512 \
    architecture.params.dropout=0.1751070468832112 \
    trainer.epochs=2 \