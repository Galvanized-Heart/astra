import os
import hydra
from omegaconf import DictConfig, OmegaConf

# It's okay to import constants here
from astra.constants import CONFIG_PATH, PROJECT_ROOT

# --- WARNING ---
# As per your original design, do NOT import torch or lightning at the top level.
# This file is the new "launcher".

@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def train(cfg: DictConfig) -> float:
    """
    The main training entry point managed by Hydra.
    """
    print("--- CONFIG LOADED ---")
    print(OmegaConf.to_yaml(cfg))

    # 1. --- Set up Deterministic Environment (from your old launcher) ---
    # This logic is critical and must run first.
    if cfg.get("seed") is not None:
        print(f"HYDRA LAUNCHER: Setting up DETERMINISTIC environment with seed: {cfg.seed}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    else:
        print("HYDRA LAUNCHER: Running in STOCHASTIC mode.")

    # 2. --- Handle File Paths ---
    # Hydra changes the working directory for each run. We need to resolve paths
    # relative to the original project root.
    # NOTE: Your old method is fine, but this is the more "Hydra-native" way.
    if hasattr(cfg.data, "train_path"):
        cfg.data.train_path = hydra.utils.to_absolute_path(cfg.data.train_path)
    if hasattr(cfg.data, "valid_path"):
        cfg.data.valid_path = hydra.utils.to_absolute_path(cfg.data.valid_path)
    
    print(f"\nTraining path {cfg.data.get('train_path')}.")
    print(f"\nValidation path {cfg.data.get('train_path')}.")

    # 3. --- Local Imports (Respecting your original design) ---
    from astra.pipelines.train_builder import PipelineBuilder

    print(f"HYDRA LAUNCHER: Handing off to pipeline builder for run...")
    
    # 4. --- Instantiate and Run the Pipeline ---
    # We pass the Hydra config object directly to the builder.
    builder = PipelineBuilder(config_dict=cfg) # We will update PipelineBuilder to accept this
    final_metric = builder.run() # No need to pass callbacks, Hydra can handle that too eventually

    print("\nTraining complete!")
    
    # Hydra can use the return value for optimization (e.g., with Optuna)
    return final_metric


if __name__ == "__main__":
    train()