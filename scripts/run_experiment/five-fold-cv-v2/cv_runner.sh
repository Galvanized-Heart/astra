#!/bin/bash
#SBATCH --job-name=cv_run      # Default job name (will be overridden by --job-name from Python script)
#SBATCH --output=slurm_logs/%x-%j.out # Standard output and error log
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # Number of tasks (processes)
#SBATCH --cpus-per-task=4      # Number of CPU cores per task
#SBATCH --mem=16G              # Memory per node
#SBATCH --time=04:00:00        # Wall clock time (e.g., 4 hours)
#SBATCH --partition=debug      # Specify your partition (e.g., debug, gpu, shared)
#SBATCH --gres=gpu:1           # Request 1 GPU (if needed)

# Create a logs directory if it doesn't exist
mkdir -p slurm_logs

echo "Starting SLURM Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

# --- Load your environment ---
# Example: Activate conda environment
# Replace 'your_conda_env' with the name of your conda environment or its path
# module load miniconda3
# source activate /path/to/your/conda_env # Or use `conda activate your_conda_env` if init'd shell

# Or if using virtualenv
# source /path/to/your/venv/bin/activate

# For Hydra, you often need to ensure the working directory is correct.
# If your main script expects to be run from the project root, ensure this is set.
# Example: cd /path/to/your/project/root

# --- Execute your main Python script with the provided overrides ---
# The arguments to this script will be the Hydra overrides.
# Hydra will automatically parse these `key=value` pairs.
# Assuming your main script is called 'main.py' and located in the project root
echo "Running command: python main.py $@"
python main.py "$@"

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo "Python script finished successfully."
else
    echo "Python script failed."
    exit 1 # Indicate failure to SLURM
fi

echo "End time: $(date)"