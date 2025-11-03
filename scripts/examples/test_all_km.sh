#!/bin/bash
#SBATCH --job-name=astra_debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2      # Adjust based on what ONE job needs
#SBATCH --mem-per-cpu=4G       # Adjust based on what ONE job needs
#SBATCH --time=00:10:00        # 10min
#SBATCH --output=slurm_logs/%j/debug_bundle.log

# Set cache
export SCRATCH_CACHE_DIR="home/maxkirby/scratch/.cache"
export WANDB_DATA_DIR="$SCRATCH_CACHE_DIR/wandb-data"
export WANDB_CACHE_DIR="$SCRATCH_CACHE_DIR/wandb"
export WANDB_CONFIG_DIR="$SCRATCH_CACHE_DIR/wandb-config"
export WANDB_DIR="$SCRATCH_CACHE_DIR/wandb-logs"      
export HF_HOME="${SCRATCH_CACHE_DIR}/huggingface"
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$HF_HOME"

export HYDRA_FULL_ERROR=1

# Define standard paths to keep commands clean
SCRIPT_PATH="/scratch/maxkirby/astra/src/astra/pipelines/hydra_train.py"

echo "Starting bundled debug job on node $(hostname)"

# CMD 1
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK uv run $SCRIPT_PATH \
    experiment_mode=single_task/km_only \
    > slurm_logs/${SLURM_JOB_ID}/run1.out 2> slurm_logs/${SLURM_JOB_ID}/run1.err &

# CMD 2
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK uv run $SCRIPT_PATH \
    experiment_mode=single_task/km_only \
    > slurm_logs/${SLURM_JOB_ID}/run2.out 2> slurm_logs/${SLURM_JOB_ID}/run2.err &

# CMD 3
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK uv run $SCRIPT_PATH \
    experiment_mode=single_task/km_only \
    > slurm_logs/${SLURM_JOB_ID}/run3.out 2> slurm_logs/${SLURM_JOB_ID}/run3.err &

# CMD 4
srun --exclusive --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK uv run $SCRIPT_PATH \
    experiment_mode=single_task/kcat_only \
    > slurm_logs/${SLURM_JOB_ID}/run4.out 2> slurm_logs/${SLURM_JOB_ID}/run4.err &

wait

echo "All sub-tasks finished."