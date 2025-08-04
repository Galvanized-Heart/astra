from astra.constants import PROJECT_ROOT
from astra.pipelines.run_train import run_training_engine

# Set path to experiment configuration
config_path = PROJECT_ROOT/"configs/experiments/test_config.yaml"

# Run training script
run_training_engine(config_path)

# NOTE: For a seeded experiment, environment variables need to be set before importing pytorch or lightning
# to ensure deterministic algorithms are used and using the cli tool allows that to happen in this example.
# unseeded experiments can call train() from astra.pipelines.train without any issues.  