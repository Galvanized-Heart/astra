from pydantic import BaseModel, FilePath, PositiveInt, Field, ValidationError, RootModel
from typing import Literal, Optional, Dict, List, Any

# Import registries to dynamically create Literals
from astra.data_processing.configs.registry import (
    MODEL_REGISTRY, LOSS_FN_REGISTRY, OPTIMIZER_REGISTRY,
    FEATURIZER_REGISTRY, SCHEDULER_REGISTRY, RECOMPOSITION_REGISTRY
)

# --- Nested Models for Organization ---

class DataConfig(BaseModel):
    train_path: FilePath
    valid_path: FilePath
    batch_size: PositiveInt = 32
    target_columns: List[str] = Field(default_factory=lambda: ["kcat", "KM", "Ki"])
    target_transform: Optional[str] = None

class FeaturizerConfig(BaseModel):
    # Dynamically get the available featurizer names
    name: Literal[tuple(FEATURIZER_REGISTRY.keys())]
    # This line now correctly defines params as a flexible dictionary
    params: Dict[str, Any] = Field(default_factory=dict)

class FeaturizersConfig(BaseModel):
    protein: FeaturizerConfig
    ligand: FeaturizerConfig

class ModelArchitectureConfig(BaseModel):
    name: Literal[tuple(MODEL_REGISTRY.keys())]
    params: Dict[str, Any] = Field(default_factory=dict)

class SchedulerConfig(BaseModel):
    name: Literal[tuple(SCHEDULER_REGISTRY.keys())]
    params: Dict[str, Any] = Field(default_factory=dict)

class LightningModuleConfig(BaseModel):
    lr: float = 1e-3
    optimizer: Literal[tuple(OPTIMIZER_REGISTRY.keys())] = "AdamW"
    loss_function: Optional[Literal[tuple(LOSS_FN_REGISTRY.keys())]] = None
    recomposition_func: Optional[Literal[tuple(RECOMPOSITION_REGISTRY.keys())]] = None
    lr_scheduler: Optional[SchedulerConfig] = None

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