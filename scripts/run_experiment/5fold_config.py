import wandb
from astra.constants import PROJECT_ROOT
from astra.pipelines.run_train import run_training_engine

for i in range(5):
    # Set path to experiment configuration
    config_path = PROJECT_ROOT/f"configs/experiments/5folds_conv/cpi_conv_model_config_fold{i}.yaml"

    # Run training script
    run_training_engine(config_path)

    # End the wandb run to start anew
    wandb.finish()