import wandb
import json
import math
from typing import List, Dict, Any, Optional, Generator
from collections import defaultdict



ENTITY = "lmse-university-of-toronto"
PROJECT = "astra"

OUTPUT_FILE = "cv_run_configs.json"

# Tags to identify and categorize runs
K_TOP_CONFIGS = 5
MODEL_TAGS = ["LinearBaselineModel", "CpiPredConvModel", "CpiPredSelfAttnModel", "CpiPredCrossAttnModel"]
PREDICTION_TAGS = {"kcat", "KM", "Ki"}
MULTI_TASK_METRICS_TO_OPTIMIZE = ["kcat", "KM", "Ki"] # Define the metrics to get top K for
SPLIT_TAGS = ["hpo_split"]
VERSION_TAG = "v2"

# Map model tags to the name used in the hydra config's architecture default
MODEL_TAG_TO_ARCH_NAME = {
    "CpiPredCrossAttnModel": "cpi_pred_cross_attn",
    "CpiPredConvModel": "cpi_pred_conv",
    "CpiPredSelfAttnModel": "cpi_pred_self_attn",
    "LinearBaselineModel": "linear",
}

# Explicitly define tunable HPO parameters and their final Hydra paths
KNOWN_HYPERPARAMETERS = {
    'lr': 'model.lightning_module.lr', 
    'batch_size': 'data.batch_size',
    'w_kcat_logit': 'model.lightning_module.loss_weights.w_kcat_logit',
    'w_km_logit': 'model.lightning_module.loss_weights.w_km_logit',
    'w_ki_logit': 'model.lightning_module.loss_weights.w_ki_logit',
    # Linear Params
    'dim_1': 'model.architecture.params.dim_1',
    'dim_2': 'model.architecture.params.dim_2',
    # Conv Params
    'hid_dim': 'model.architecture.params.hid_dim',
    'kernal_1': 'model.architecture.params.kernal_1',
    'conv_out_dim': 'model.architecture.params.conv_out_dim',
    'kernal_2': 'model.architecture.params.kernal_2',
    'last_hid': 'model.architecture.params.last_hid',
    'dropout': 'model.architecture.params.dropout',
    # Attention Params
    'n_heads': 'model.architecture.params.n_heads',
    'd_ff': 'model.architecture.params.d_ff',
    'd_k': 'model.architecture.params.d_k',
    'd_v': 'model.architecture.params.d_v',
    'attn_out_dim': 'model.architecture.params.attn_out_dim',
}



def find_value_in_nested_dict(key_to_find: str, nested_dict: Dict[str, Any]) -> Optional[Any]:
    """Recursively searches for a key in a nested dictionary."""
    if key_to_find in nested_dict:
        return nested_dict[key_to_find]
    for value in nested_dict.values():
        if isinstance(value, dict):
            if (found_value := find_value_in_nested_dict(key_to_find, value)) is not None:
                return found_value
    return None



def extract_hpo_overrides(wandb_config: Dict[str, Any]) -> Dict[str, Any]:
    """Selectively extracts known hyperparameters from a wandb config."""
    overrides = {}
    for param_name, hydra_path in KNOWN_HYPERPARAMETERS.items():
        found_value = None
        
        # Look for nested key-value pair
        found_value = find_value_in_nested_dict(param_name, wandb_config)
        
        # If not assigned in a nested manner, look for nest as if it were a full key
        if found_value is None and hydra_path in wandb_config:
            found_value = wandb_config[hydra_path]
        
        if found_value is not None:
            overrides[hydra_path] = found_value
            
    return overrides



def categorize_run(run: wandb.apis.public.Run) -> Generator[Dict[str, Any], None, None]:
    """
    Analyzes a run's tags and yields one or more categorization profiles.
    A multi-task run can yield a profile for each target metric (kcat, KM, Ki).
    """
    run_tags = set(run.tags)
    found_pred_tags = run_tags.intersection(PREDICTION_TAGS)
    
    recomposition = "none"
    if "BasicRecomp" in run_tags: recomposition = "basic"
    elif "AdvancedRecomp" in run_tags: recomposition = "advanced"

    base_info = {
        "config": {k: v for k, v in run.config.items() if not k.startswith('_')},
        "run_id": run.id
    }

    # Case 1: Single-Task Run
    if len(found_pred_tags) == 1:
        target_param = found_pred_tags.pop()
        metric_name = f"valid/{target_param}_Pearson"
        metric_value = run.summary.get(metric_name)
        
        if isinstance(metric_value, (float, int)) and not math.isnan(metric_value):
            yield {
                **base_info, "run_type": "single_task", "target_param": target_param,
                "metric_value": metric_value, "optimized_for": target_param, 
                "recomposition": recomposition
            }

    # Case 2: Multi-Task Run
    elif len(found_pred_tags) >= 2:
        # For a multi-task run, check each potential metric we want to optimize for
        for metric in MULTI_TASK_METRICS_TO_OPTIMIZE:
            metric_name = f"valid/{metric}_Pearson"
            metric_value = run.summary.get(metric_name)
            
            if isinstance(metric_value, (float, int)) and not math.isnan(metric_value):
                yield {
                    **base_info, "run_type": "multi_task", "target_param": "all",
                    "metric_value": metric_value, "optimized_for": metric,
                    "recomposition": recomposition
                }



def main():
    api = wandb.Api()

    # defaultdict for organizing runs by top k rankings
    all_generated_configs_by_rank = defaultdict(list)

    print("--- Generating Cross-Validation Run Definitions ---")
    
    for model_tag in MODEL_TAGS:
        arch_name = MODEL_TAG_TO_ARCH_NAME.get(model_tag)

        # Skip if arch isn't in list
        if not arch_name: 
            print(f"  --> Warning: No architecture name mapping found for model tag '{model_tag}'. Skipping.")
            continue

        # Set filters
        grouped_runs = defaultdict(list)
        filters = {"$and": [
            {"tags": model_tag}, 
            {"tags": {"$in": SPLIT_TAGS}}, 
            {"tags": VERSION_TAG}
        ]}
        print(f"\nQuerying project '{PROJECT}' for model: '{model_tag}' (filtered by '{VERSION_TAG}' and HPO splits)...") 
        
        try:
            # Fetch runs. The api.runs() returns an iterator.
            runs_iterator = api.runs(f"{ENTITY}/{PROJECT}", filters=filters)
            
            # Convert to list to iterate multiple times if needed, and to get total count
            # Be careful with very large numbers of runs, though for HPO this is usually fine.
            runs = list(runs_iterator) 
            print(f"  --> Found {len(runs)} potential runs for '{model_tag}' with filters.")
        except Exception as e:
            print(f"  --> An error occurred while fetching runs for '{model_tag}': {e}. Skipping."); continue

        if not runs:
            print(f"  --> No runs found for model '{model_tag}' after initial filter for '{VERSION_TAG}'.")
            continue

        # Get valid run configs from the fetched runs
        found_valid_run_configs = 0
        for run in runs:
            for run_info in categorize_run(run):
                # Ensure run_info contains 'recomposition' (addressed in categorize_run fix)
                if 'recomposition' not in run_info:
                    print(f"  --> Warning: Run {run.id} missing 'recomposition' info after categorization. Skipping.")
                    continue
                
                category_key = (
                    f"{run_info['run_type']}-"
                    f"{run_info['target_param']}-"
                    f"{run_info['recomposition']}-"
                    f"opt_{run_info['optimized_for']}"
                )
                # Store (metric_value, config, run_id) for better traceability
                grouped_runs[category_key].append((run_info['metric_value'], run_info['config'], run_info['run_id']))
                found_valid_run_configs += 1
        
        if not grouped_runs:
            print(f"  --> No valid HPO runs found for model '{model_tag}' after categorization (e.g., no valid metrics).") 
            continue
        else:
            print(f"  --> Categorized {found_valid_run_configs} valid run configs for '{model_tag}'.")

        # Process grouped runs to find top K
        for category_key, runs_data in grouped_runs.items():
            print(f"  Processing category: {category_key} ({len(runs_data)} valid runs found)")

            # Sort runs by metric_value in descending order to get top K
            runs_data.sort(key=lambda x: x[0], reverse=True)
            
            # Extract top K configs (and their original run_ids for reference)
            top_k_items = runs_data[:K_TOP_CONFIGS]
            
            # Unpack the more detailed category key
            # Example: "single_task-kcat-none-opt_kcat"
            parts = category_key.replace("opt_", "").split('-')
            if len(parts) == 4:
                run_type, target, recomp, opt_metric = parts
            else:
                print(f"  --> Warning: Malformed category key '{category_key}'. Skipping.")
                continue

            for i, (metric_val, config, original_run_id) in enumerate(top_k_items): 
                # `i` here represents the rank: 0 for top 1, 1 for top 2, etc.
                for fold_index in range(5): # Generate configurations for 5-fold cross-validation
                    target_columns_override = f"{target.lower()}_only" if run_type == 'single_task' else "all"
                    
                    run_overrides = {
                        "architecture": arch_name,
                        "data": f"fold_{fold_index}",
                        "target_columns": target_columns_override,
                        "recomposition": recomp,
                        "wandb.group": f"{model_tag}-{target}-{recomp}-valid/{opt_metric}_Pearson-top{i}",
                    }

                    hpo_overrides = extract_hpo_overrides(config)
                    run_overrides.update(hpo_overrides)
                    all_generated_configs_by_rank[i].append(run_overrides)

    # Order master list by top k ranks
    master_run_list = []
    for rank in range(K_TOP_CONFIGS):
        master_run_list.extend(all_generated_configs_by_rank[rank])

    # Generate json for master run
    print(f"\nGenerated a total of {len(master_run_list)} run configurations.")
    if not master_run_list:
        print("No configurations generated. Please check your WandB project, tags, and data.")
    else:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(master_run_list, f, indent=2)
        print(f"Master run override list saved to '{OUTPUT_FILE}'")



if __name__ == "__main__":
    main()