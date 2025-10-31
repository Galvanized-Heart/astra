#!/bin/bash
#SBATCH --job-name="hpo-job"
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=10:00:00
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

uv run src/astra/pipelines/hydra_train.py \
    architecture=cpi_pred_self_attn \
    experiment_mode=multi_task/basic \
    architecture.params.d_ff=64 \
    architecture.params.d_k=512 \
    architecture.params.d_v=256 \
    architecture.params.n_heads=16 \
    model.lightning_module.loss_weights.w_kcat_logit=1.5725465872364937 \
    model.lightning_module.loss_weights.w_ki_logit=0.8184742142309025 \
    model.lightning_module.loss_weights.w_km_logit=-1.409807268527442 \
    model.lightning_module.lr=0.0029945662157201745 \
    trainer.epochs=10 \
    data.batch_size=155