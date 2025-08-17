import wandb
import argparse
import json
import yaml
import copy
from astra.pipelines.run_train import run_training_engine_from_dict
from astra.constants import PROJECT_ROOT

def recursive_update(base_dict, new_dict):
    """Recursively updates a dictionary."""
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def main():
    parser = argparse.ArgumentParser(description="Run a single training fold based on a master run list.")
    parser.add_argument("--run_id", type=int, required=True, help="The ID of the run to execute from the master list.")
    args = parser.parse_args()
    
    run_id = args.run_id

    # Load the master list and find our specific instructions
    master_list_path = PROJECT_ROOT / "scripts/run_experiment/five-fold-cv/master_run_list.json"
    with open(master_list_path, 'r') as f:
        run_definition = json.load(f)[run_id]

    print(f"--- EXECUTING RUN ID: {run_id} ---")
    print(json.dumps(run_definition, indent=2))

    base_config_path = PROJECT_ROOT / "configs/experiments/five-fold-cv/base_template_config.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # --- Build the final config for this specific run ---
    final_config = copy.deepcopy(base_config)
    final_config = recursive_update(final_config, run_definition['hyperparameters'])
    
    final_config['model']['architecture']['name'] = run_definition['arch']
    
    if run_definition['task_type'] == 'single_task':
        final_config['data']['target_columns'] = [run_definition['target_param']]
    else: # Multi-task
        final_config['data']['target_columns'] = ["kcat", "KM", "Ki"]
        if run_definition['task_type'] == 'basic_recomp':
            final_config['model']['lightning_module']['recomposition_func'] = "BasicRecomp"
        elif run_definition['task_type'] == 'advanced_recomp':
            final_config['model']['lightning_module']['recomposition_func'] = "AdvancedRecomp"

    fold_index = run_definition['fold_index']
    final_config['data']['train_path'] = f"data/cv_splits_pangenomic/fold_{fold_index}_train.csv"
    final_config['data']['valid_path'] = f"data/cv_splits_pangenomic/fold_{fold_index}_valid.csv"
    
    # --- Add W&B info directly into the config for the PipelineBuilder ---
    final_config['wandb'] = {
        'group': run_definition['group_name'],
        'name': run_definition['run_name'],
        'tags': run_definition['tags'],
        'job_type': "cv-fold-train"
    }
    
    # Launch the training. This will create its own fresh wandb run.
    run_training_engine_from_dict(config_dict=final_config)

if __name__ == "__main__":
    main()