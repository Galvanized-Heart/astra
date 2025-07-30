from pathlib import Path
import subprocess

from astra.constants import PROJECT_ROOT


train_path = valid_path = Path.joinpath(PROJECT_ROOT, "data", "split", "test.csv")
epochs = 10
batch_size = 32
seed = 42

# Run training script
command = ["uv", "run", "astra", "train", "--epochs", "15", "--seed", "42", "--batch_size", "32", "--train_path", "../../data/split/train.csv"]
subprocess.run(command)