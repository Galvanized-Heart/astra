import yaml
import os
from tqdm import tqdm

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

    print(f"Setting up training for {config.get("train_path")}.")
    print(f"Using {config.get("valid_path")} for validation.")

    # Establish whether or not to use deterministic algorithms
    # WARNING: This must happen torch or lightning is imported directly or indirectly!
    seed = config.get("seed")
    if seed is not None:
        print(f"LAUNCHER: Setting up DETERMINISTIC environment with seed: {seed}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Or ":16:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        print("LAUNCHER: Running in STOCHASTIC mode.")

    try:
        # WARNING: Do not import torch or lightning into run_train.py directly or indirectly before env vars are set!
        from astra.data_processing.configs.config_schema import FullConfig, ValidationError

        # Parse and validate the raw dictionary
        config = FullConfig(**config_dict)
        print("Configuration validated successfully!")

    except ValidationError as e:
        # On failure, print a user-friendly error and exit.
        print("\nERROR: Configuration validation failed!")
        print(e)
        exit

    # Import locally after environment variable setup
    with tqdm(total=1, desc="Loading libraries") as pbar:
        # WARNING: Do not import torch or lightning into run_train.py directly or indirectly before env vars are set!
        from astra.pipelines.train import train
        pbar.update(1)
    
    # Run training logic
    print(f"LAUNCHER: Handing off to training engine for run '{config.get('run_name')}'...")
    train(config_dict)
    print("\nTraining complete!")