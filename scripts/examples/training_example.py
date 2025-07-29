from pathlib import Path

from astra.pipelines.train import train
from astra.constants import PROJECT_ROOT


train_path = valid_path = Path.joinpath(PROJECT_ROOT, "data", "split", "cpipred", "pangenomic", "mmseqs", "train.csv")
batch_size = 32
seed = 42

# Run training script
train(train_path, valid_path, batch_size, seed)