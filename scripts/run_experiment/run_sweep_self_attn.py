import yaml
import wandb
import math
import argparse
from pathlib import Path
from astra.constants import PROJECT_ROOT
from astra.pipelines.run_train import run_training_engine_from_dict

"""
To be able to run this script using wandb sweep, input the following command into the terminal:
`uv run wandb sweep <CONFIG_PATH>`
    - Example CONFIG_PATH = configs/hpo/conv/<RECOMP TYPE>/hpo_conv_config.yaml


This will provide the user with an large output containing the <SWEEP_ID> for the agent to run:
`uv run wandb agent --count 50 <SWEEP_ID>`
    - Example SWEEP_ID = lmse-university-of-toronto/astra-scripts_run_experiment/u8m6cp00


NOTE: count=50 is the run budget for the HPO experiments and this must be maintained across 
different models and recomp functions.
"""

def softmax(logits: list) -> list:
    """
    Calculates softmax in pure Python to avoid importing torch at the top level.
    """
    # Subtract max for numerical stability (prevents overflow with large logits)
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    
    sum_exps = sum(exps)
    softmax_probs = [e / sum_exps for e in exps]
    
    return softmax_probs

def main():
    """This function is called by the wandb agent for each trial."""
    
    parser = argparse.ArgumentParser(description="Run a W&B sweep trial.")
    parser.add_argument(
        "--base_config", 
        type=str, 
        required=True, 
        help="Path to the base YAML configuration file, relative to project root."
    )
    args, _ = parser.parse_known_args()    
    
    run = wandb.init()
    
    # Use the path from the command line, resolved from the project root
    base_config_path = PROJECT_ROOT / args.base_config
    with open(base_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Masked loss function weightings
    logits = [wandb.config.w_kcat_logit, wandb.config.w_km_logit, wandb.config.w_ki_logit]
    final_weights = softmax(logits)
    config_dict['model']['lightning_module']['loss_function']['params']['weights'] = final_weights

    # Batch size
    config_dict['data']['batch_size'] = wandb.config.batch_size

    # Model architecture
    config_dict['model']['architecture']['params']['n_heads'] = wandb.config.n_heads
    config_dict['model']['architecture']['params']['d_k'] = wandb.config.d_k
    config_dict['model']['architecture']['params']['d_v'] = wandb.config.d_v
    config_dict['model']['architecture']['params']['d_ff'] = wandb.config.d_ff

    # Learning rate
    config_dict['model']['lightning_module']['lr'] = wandb.config.lr

    # 4. Run the training with the merged configuration
    # The training engine will log metrics to W&B automatically. No return value needed.
    run_training_engine_from_dict(config_dict=config_dict)

if __name__ == "__main__":
    main()