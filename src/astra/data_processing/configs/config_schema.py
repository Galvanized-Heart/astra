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
    # This allows any parameters, you could make it stricter if needed
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
    # Dynamically create the Literal from registered models
    name: Literal[Tuple(MODEL_REGISTRY.registered_names)]
    params: Dict[str, Any] = Field(default_factory=dict)

class LightningModuleConfig(BaseModel):
    lr: float = 1e-3
    optimizer: Literal[Tuple(OPTIMIZER_REGISTRY.registered_names)] = "AdamW"
    loss_function: Literal[Tuple(LOSS_FN_REGISTRY.registered_names)] = "MaskedMSELoss"

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

# --- Top-Level Configuration Model ---

class FullConfig(BaseModel):
    """The complete, validated configuration for an Astra training run."""
    project_name: str = "astra"
    run_name: str
    seed: Optional[int] = None

    data: DataConfig
    featurizers: FeaturizersConfig
    model: ModelConfig
    trainer: TrainerConfig