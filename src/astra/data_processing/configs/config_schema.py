from pydantic import BaseModel, FilePath, PositiveInt, Field, ValidationError
from typing import Literal, Optional, Dict, Tuple, Any

# Assuming you have a registry for your models.
# If not, you would manually define the Literal, e.g., Literal["DummyModel", "RealModel"]
from astra.data_processing.configs.registry import MODEL_REGISTRY, LOSS_FN_REGISTRY, OPTIMIZER_REGISTRY

# --- Nested Models for Organization ---

class DataConfig(BaseModel):
    train_path: FilePath  # Validates that the file path exists
    valid_path: FilePath
    batch_size: PositiveInt = 32 # Validates that the number is > 0

class FeaturizerParams(BaseModel):
    #__root__: Dict[str, Any]
    pass

class ProteinFeaturizerConfig(BaseModel):
    name: Literal["ESMFeaturizer"] # Add more as you create them
    params: Dict[str, Any] = Field(default_factory=dict)

class LigandFeaturizerConfig(BaseModel):
    name: Literal["MorganFeaturizer"] # Add more as you create them
    params: Dict[str, Any] = Field(default_factory=dict)

class FeaturizersConfig(BaseModel):
    protein: ProteinFeaturizerConfig
    ligand: LigandFeaturizerConfig

class ModelArchitectureConfig(BaseModel):
    name: Literal[tuple(MODEL_REGISTRY.registered_names)] # Dynamic literal from registry.py
    params: Dict[str, Any] = Field(default_factory=dict)

class LightningModuleConfig(BaseModel):
    lr: float = 1e-3
    optimizer: Literal[tuple(OPTIMIZER_REGISTRY.registered_names)] = "AdamW" # Dynamic literal from registry.py
    loss_function: Literal[tuple(LOSS_FN_REGISTRY.registered_names)] = "MaskedMSELoss" # Dynamic literal from registry.py

class ModelConfig(BaseModel):
    architecture: ModelArchitectureConfig
    lightning_module: LightningModuleConfig

class CheckpointCallbackConfig(BaseModel):
    monitor: str = "valid_loss_epoch"
    save_top_k: PositiveInt = 1
    mode: Literal["min", "max"] = "min"

class CallbacksConfig(BaseModel):
    checkpoint: CheckpointCallbackConfig

class TrainerConfig(BaseModel):
    epochs: PositiveInt = 10
    device: str = "auto"
    callbacks: CallbacksConfig



class FullConfig(BaseModel):
    """The complete, validated configuration for an Astra training run."""
    project_name: str = "astra"
    run_name: str
    seed: Optional[int] = None

    data: DataConfig
    featurizers: FeaturizersConfig
    model: ModelConfig
    trainer: TrainerConfig