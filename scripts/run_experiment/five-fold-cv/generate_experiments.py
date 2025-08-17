# FILE: generate_experiments.py (Final Definitive Version)
import wandb
import json
import numpy as np

# --- Core Configuration ---
K_TOP_CONFIGS = 5
ENTITY = "lmse-university-of-toronto"

ARCHITECTURES = {
    "linear": "astra-linear",
    "conv": "astra",
    "self_attn": "astra-self-attn",
    "cross_attn": "astra-cross-attn"
}
ARCH_TO_MODEL_NAME_MAP = {
    "linear": "LinearBaselineModel",
    "conv": "CpiPredConvModel",
    "self_attn": "CpiPredSelfAttnModel",
    "cross_attn": "CpiPredCrossAttnModel"
}
# Use lowercase for project names
KINETIC_PARAMS_LOWER = ["kcat", "km", "ki"]
# Map lowercase to the correctly cased metric names found in W&B
METRIC_NAME_MAP = {
    "kcat": "kcat_Pearson",
    "km": "KM_Pearson",
    "ki": "Ki_Pearson"
}
PARAM_TO_COLUMN_NAME_MAP = {
    "kcat": "kcat",
    "km": "KM",
    "ki": "Ki"
}
MULTI_TASK_TYPES = {
    "direct": "multi-task-no-recomp-hpo",
    "basic_recomp": "multi-task-basic-recomp-hpo",
    "advanced_recomp": "multi-task-advanced-recomp-hpo"
}



def structure_hyperparameters(flat_params: dict) -> dict:
    """
    Organizes a flat dictionary of hyperparameters into the nested structure
    expected by the base configuration file.
    """
    structured_params = {
        'data': {},
        'model': {'architecture': {'params': {}}, 'lightning_module': {}},
    }
    
    # Define where each parameter should go
    # This is the key logic that was missing
    mapping = {
        'batch_size': ['data', 'batch_size'],
        'lr': ['model', 'lightning_module', 'lr'],
        # Linear Model Params
        'dim_1': ['model', 'architecture', 'params', 'dim_1'],
        'dim_2': ['model', 'architecture', 'params', 'dim_2'],
        # Conv Model Params
        'hid_dim': ['model', 'architecture', 'params', 'hid_dim'],
        'kernal_1': ['model', 'architecture', 'params', 'kernal_1'],
        'conv_out_dim': ['model', 'architecture', 'params', 'conv_out_dim'],
        'kernal_2': ['model', 'architecture', 'params', 'kernal_2'],
        'last_hid': ['model', 'architecture', 'params', 'last_hid'],
        'dropout': ['model', 'architecture', 'params', 'dropout'],
        # Attention Model Params
        'n_heads': ['model', 'architecture', 'params', 'n_heads'],
        'd_ff': ['model', 'architecture', 'params', 'd_ff'],
        'd_k': ['model', 'architecture', 'params', 'd_k'],
        'd_v': ['model', 'architecture', 'params', 'd_v'],
        'attn_out_dim': ['model', 'architecture', 'params', 'attn_out_dim'],
        # NOTE: Add any other swept params here following the pattern
    }

    for key, value in flat_params.items():
        if key in mapping:
            path = mapping[key]
            # Navigate the nested dict and place the value
            current_level = structured_params
            for p in path[:-1]:
                current_level = current_level[p]
            current_level[path[-1]] = value
            
    return structured_params



def get_top_k_configs(api, project_name, metric, k):
    """Queries a wandb project and returns the top k hyperparameter configs."""
    print(f"  Querying project: {project_name} for metric: {metric}")
    try:
        runs = api.runs(f"{ENTITY}/{project_name}")
    except wandb.errors.CommError:
        print(f"  --> Project not found. Skipping.")
        return []

    valid_runs = []
    for run in runs:
        if run.state != "finished" or metric not in run.summary:
            continue
        metric_value = run.summary.get(metric)
        if not isinstance(metric_value, (int, float, np.number)):
            continue
        valid_runs.append(run)

    valid_runs.sort(key=lambda run: run.summary[metric], reverse=True)
    
    top_configs = []
    for run in valid_runs[:k]:
        flat_params = {k: v for k, v in run.config.items() if not k.startswith('_')}
        structured_params = structure_hyperparameters(flat_params) # Convert to nested dict
        top_configs.append(structured_params)
        
    if len(top_configs) < k:
        print(f"  --> WARNING: Found only {len(top_configs)} valid runs out of {len(runs)} total that match criteria.")
    return top_configs




def main():
    api = wandb.Api()
    master_run_list = []
    run_id = 0

    # --- THIS IS THE KEY CHANGE ---
    # We now loop through folds here to create a flat list of all 600 runs.
    
    print("--- Generating Single-Task Run Definitions ---")
    for arch_key, arch_prefix in ARCHITECTURES.items():
        for param_lower in KINETIC_PARAMS_LOWER:
            project_name = f"{arch_prefix}-single-task-{param_lower}-hpo"
            metric_base_name = METRIC_NAME_MAP[param_lower]
            metric = f"valid/{metric_base_name}"
            
            top_configs = get_top_k_configs(api, project_name, metric, K_TOP_CONFIGS)
            for i, config in enumerate(top_configs):
                for fold_index in range(5): # <-- NESTED LOOP FOR FOLDS
                    tags = [
                        "final-cv", arch_key, "single_task",
                        f"target_{param_lower.upper()}", f"top_{i}", f"fold_{fold_index}"
                    ]
                    model_name = ARCH_TO_MODEL_NAME_MAP[arch_key]
                    column_name = PARAM_TO_COLUMN_NAME_MAP[param_lower]
                    
                    master_run_list.append({
                        "run_id": run_id,
                        "arch": model_name,
                        "task_type": "single_task",
                        "target_param": column_name,
                        "fold_index": fold_index,
                        "hyperparameters": config,
                        "group_name": f"{arch_key}-st-{param_lower}-top{i}",
                        "run_name": f"fold-{fold_index}",
                        "tags": tags
                    })
                    run_id += 1

    print("\n--- Generating Multi-Task Run Definitions ---")
    for arch_key, arch_prefix in ARCHITECTURES.items():
        for mt_key, mt_suffix in MULTI_TASK_TYPES.items():
            project_name = f"{arch_prefix}-{mt_suffix}"
            metric = "valid/Ki_Pearson"
            
            top_configs = get_top_k_configs(api, project_name, metric, K_TOP_CONFIGS)
            for i, config in enumerate(top_configs):
                for fold_index in range(5): # <-- NESTED LOOP FOR FOLDS
                    tags = [
                        "final-cv", arch_key, "multi_task",
                        mt_key, f"top_{i}", f"fold_{fold_index}"
                    ]
                    model_name = ARCH_TO_MODEL_NAME_MAP[arch_key]
                    master_run_list.append({
                        "run_id": run_id,
                        "arch": model_name,
                        "task_type": mt_key,
                        "target_param": "all",
                        "fold_index": fold_index,
                        "hyperparameters": config,
                        "group_name": f"{arch_key}-mt-{mt_key}-top{i}",
                        "run_name": f"fold-{fold_index}",
                        "tags": tags
                    })
                    run_id += 1

    print(f"\nGenerated a total of {len(master_run_list)} run configurations.")
    output_path = "master_run_list.json"
    with open(output_path, 'w') as f:
        json.dump(master_run_list, f, indent=2)
    print(f"Master run list saved to '{output_path}'")

if __name__ == "__main__":
    main()