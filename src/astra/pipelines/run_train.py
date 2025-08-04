import yaml
import os

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

    # Validate config
    try:
        # WARNING: Do not import torch or lightning into run_train.py directly or indirectly before env vars are set!
        from astra.data_processing.configs.config_schema import FullConfig, ValidationError

        # Parse and validate the raw dictionary
        validated_config = FullConfig(**config_dict)
        print("Configuration validated successfully!")

    except ValidationError as e:
        # On failure, print a user-friendly error and exit.
        print("\nERROR: Configuration validation failed!")
        print(e)
        exit

    # Import locally after environment variable setup
    # WARNING: Do not import torch or lightning into run_train.py directly or indirectly before env vars are set!
    from astra.pipelines.train_builder import PipelineBuilder

    print(f"LAUNCHER: Handing off to pipeline builder for run '{validated_config.run_name}'...")

    # Run training logic    
    builder = PipelineBuilder(config=validated_config)
    builder.run()

    print("\nTraining complete!")