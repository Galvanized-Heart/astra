from pathlib import Path

from astra.pipelines.train import train
from astra.pipelines.train_xgboost import train_xgboost
from astra.constants import PROJECT_ROOT


train_path = valid_path = Path.joinpath(PROJECT_ROOT, "data", "split", "train.csv")
batch_size = 32
seed = 42

# Run training script
train(train_path, valid_path, batch_size, seed)

# Run XGBoost training script
train_xgboost(train_path, valid_path, seed)