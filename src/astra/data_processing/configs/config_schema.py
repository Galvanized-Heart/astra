from pydantic import BaseModel, FilePath, ValidationError
from typing import Literal, Optional

class DataConfig(BaseModel):
    train_path: FilePath  # Pydantic validates that the file exists!
    valid_path: FilePath
    batch_size: int = 32

class ModelConfig(BaseModel):
    name: Literal["DummyModel", "RealModel"] # Only allows these values
    params: dict = {}

class TrainerConfig(BaseModel):
    epochs: int
    # ...

class FullConfig(BaseModel):
    seed: Optional[int] = 42
    run_name: str
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
    # ... and so on

# In run_train, you would do this at the very beginning:
def run_train(config: dict):
    try:
        # Validate and structure the raw dict
        validated_config = FullConfig(**config)
    except ValidationError as e:
        print(f"Configuration error: {e}")
        return

    # Now you can use validated_config with dot notation, e.g.,
    # validated_config.data.batch_size