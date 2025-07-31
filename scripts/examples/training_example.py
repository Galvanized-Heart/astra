from pathlib import Path
import subprocess

from astra.constants import PROJECT_ROOT

# Run training script
command = [
    "uv", "run", "astra", "train", 
    "--epochs", "15", 
    "--seed", "42", # If you want to run in stochastic mode, do not set a seed
    "--batch_size", "32", 
    "--train_path", f"{PROJECT_ROOT}/data/split/train.csv",
    "--valid_path", f"{PROJECT_ROOT}/data/split/valid.csv"
]
subprocess.run(command)

# NOTE: For a seeded experiment, environment variables need to be set before importing pytorch or lightning
# to ensure deterministic algorithms are used and using the cli tool allows that to happen in this example.
# unseeded experiments can call train() from astra.pipelines.train without any issues.  