#!/bin/bash

# NOTE: All #SBATCH directives have been removed from this script.
# The hpo_manager.sh script is now responsible for requesting resources.

# This script is a simple worker. It expects two arguments:
# 1. SWEEP_ID (required)
# 2. COUNT (optional, defaults to 1)

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed. Helps debug.
set -x

# Function to be called on exit or error
cleanup() {
    echo "--- Agent worker script finished at $(date) with exit code $? ---"
}
# Trap EXIT signal to run cleanup function always
trap cleanup EXIT

if [ -z "$1" ]; then
    echo "ERROR: Missing argument. This script requires a Sweep ID."
    exit 1
fi

SWEEP_ID=$1
COUNT=${2:-1}

echo "Starting agent worker for Sweep ID: ${SWEEP_ID}"
echo "This agent will run for ${COUNT} trial(s)."

echo "Running on node: $(hostname)"
echo "--- System Memory ---"
free -h
echo "--- Disk Space on /tmp and Filesystem ---"
df -h /tmp
df -h .

export SCRATCH_CACHE_DIR="/gpfs/fs0/scratch/m/mahadeva/maxkirby/.cache"
export WANDB_DATA_DIR="$SCRATCH_CACHE_DIR/wandb-data"
export WANDB_CACHE_DIR="$SCRATCH_CACHE_DIR/wandb"
export WANDB_CONFIG_DIR="$SCRATCH_CACHE_DIR/wandb-config"
export WANDB_DIR="$SCRATCH_CACHE_DIR/wandb-logs"      
export HF_HOME="${SCRATCH_CACHE_DIR}/huggingface"
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$HF_HOME"

# Change to the project directory to ensure all paths are correct
cd "/gpfs/fs0/scratch/m/mahadeva/maxkirby/astra" || exit 1

# Force single-GPU mode
export CUDA_VISIBLE_DEVICES=0

# --- Run the agent for exactly one trial ---
# uv run will use the ./.venv in the current directory
uv run wandb agent --count "${COUNT}" "${SWEEP_ID}"

echo "Agent worker finished successfully after ${COUNT} trial(s)."