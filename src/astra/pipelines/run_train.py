import yaml
import os
import sys
from astra.constants import PROJECT_ROOT

################################
########!!! WARNING !!!#########
################################
###  DO NOT IMPORT torch OR  ###
### lightning INTO THIS FILE ###
################################
###  DOING SO WILL RUIN THE  ###
###       DETERMINISTIC      ###
###       FUNCTIONALIY       ###
################################

def run_training_engine(config_path):
    """
    Base function for training Astra model.
    
    Loads yaml config file, establishes deterministic algorithms if the run is seeded, and runs the training logic.
    """

    print(f"Reading config file from {config_path}")

    # Read the yaml config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    final_metric = run_training_engine_from_dict(config_dict=config_dict)

    return final_metric


def run_training_engine_from_dict(config_dict):
    """
    Training function configured to handle a config dictionary object directly.
    """

    print("INFO: Resolving relative paths in config against PROJECT_ROOT...")
    # Resolve train_path
    if 'data' in config_dict and 'train_path' in config_dict['data']:
        relative_train_path = config_dict['data']['train_path']
        # Use your constant to create an absolute path
        config_dict['data']['train_path'] = str(PROJECT_ROOT / relative_train_path)

    # Resolve valid_path
    if 'data' in config_dict and 'valid_path' in config_dict['data']:
        relative_valid_path = config_dict['data']['valid_path']
        config_dict['data']['valid_path'] = str(PROJECT_ROOT / relative_valid_path)

    data_config = config_dict.get("data")
    print(f"Setting up training for {data_config.get('train_path')}.")
    print(f"Using {data_config.get('valid_path')} for validation.")

    # Establish whether or not to use deterministic algorithms
    # WARNING: This must happen torch or lightning is imported directly or indirectly!
    seed = config_dict.get("seed")
    if seed is not None:
        print(f"LAUNCHER: Setting up DETERMINISTIC environment with seed: {seed}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Or ":16:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        print("LAUNCHER: Running in STOCHASTIC mode.")

    trainer_callbacks_config = config_dict.get("trainer", {}).get("callbacks", {})
    pruning_callback = trainer_callbacks_config.pop("pruning", None)
    
    extra_callbacks = []
    if pruning_callback:
        extra_callbacks.append(pruning_callback)


    # Import locally after environment variable setup
    # WARNING: Do not import torch or lightning into run_train.py directly or indirectly before env vars are set!
    from astra.pipelines.train_builder import PipelineBuilder

    print(f"LAUNCHER: Handing off to pipeline builder for run...")

    # Run training logic    
    builder = PipelineBuilder(config_dict=config_dict)
    final_metric = builder.run(extra_callbacks=extra_callbacks)

    print("\nTraining complete!")
    return final_metric