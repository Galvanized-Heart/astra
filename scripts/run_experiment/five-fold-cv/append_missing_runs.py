# FILE: scripts/run_experiment/five-fold-cv/append_missing_runs.py
import wandb
import json
import numpy as np
from tqdm import tqdm

# Import the helper functions from your main generation script
from generate_experiments import get_top_k_configs, ARCHITECTURES, ARCH_TO_MODEL_NAME_MAP, KINETIC_PARAMS_LOWER, METRIC_NAME_MAP, MULTI_TASK_TYPES

def main():
    api = wandb.Api()
    
    # --- Step 1: Load the existing master list ---
    existing_list_path = "master_run_list copy.json"
    try:
        with open(existing_list_path, 'r') as f:
            master_run_list = json.load(f)
        print(f"Loaded {len(master_run_list)} existing run configurations from '{existing_list_path}'.")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{existing_list_path}'. Please ensure it's in the same directory.")
        return

    newly_generated_runs = []
    
    # Define only the metrics that were missed for the multi-task experiments
    missing_metrics_params = ["kcat", "km"] # We already have 'ki'

    print("\n--- Generating MISSING Multi-Task Run Definitions ---")
    for arch_key, arch_prefix in ARCHITECTURES.items():
        for mt_key, mt_suffix in MULTI_TASK_TYPES.items():
            project_name = f"{arch_prefix}-{mt_suffix}"
            
            # Loop ONLY through the missing optimization metrics
            for param_lower in missing_metrics_params:
                metric_base_name = METRIC_NAME_MAP[param_lower]
                metric = f"valid/{metric_base_name}"
                
                print(f"\n-- Selecting top configs for {project_name} based on {metric} --")
                top_configs = get_top_k_configs(api, project_name, metric, 5)
                
                for i, config in enumerate(top_configs):
                    for fold_index in range(5):
                        tags = [
                            "final-cv", arch_key, "multi_task", mt_key,
                            f"target_{param_lower.upper()}", f"top_{i}", f"fold_{fold_index}"
                        ]
                        model_name = ARCH_TO_MODEL_NAME_MAP[arch_key]
                        group_name = f"{arch_key}-mt-{mt_key}-{param_lower}-top{i}"
                        
                        newly_generated_runs.append({
                            # We'll fix the run_id later
                            "arch": model_name,
                            "task_type": mt_key,
                            "target_param": "all",
                            "fold_index": fold_index,
                            "hyperparameters": config,
                            "group_name": group_name,
                            "run_name": f"fold-{fold_index}",
                            "tags": tags
                        })

    print(f"\nGenerated {len(newly_generated_runs)} new run configurations.")
    
    # --- Step 2: Append the new runs to the existing list ---
    updated_master_list = master_run_list + newly_generated_runs
    
    # --- Step 3: Re-index the entire list to ensure run_id is sequential ---
    for idx, run_def in enumerate(updated_master_list):
        run_def['run_id'] = idx

    print(f"The updated master list now contains a total of {len(updated_master_list)} run configurations.")
    
    # --- Step 4: Save the new, combined list ---
    output_path = "master_run_list.json" # Overwrite the old file
    with open(output_path, 'w') as f:
        json.dump(updated_master_list, f, indent=2)
        
    print(f"Successfully updated '{output_path}'.")

if __name__ == "__main__":
    main()