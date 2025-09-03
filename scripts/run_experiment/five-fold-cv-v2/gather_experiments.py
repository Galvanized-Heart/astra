import wandb
import json
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict

# --- Core Configuration ---
K_TOP_CONFIGS = 5
ENTITY = "lmse-university-of-toronto"
PROJECT = "astra"
OUTPUT_FILE = "cv_run_configs.json"

# Tags to identify and categorize runs
MODEL_TAGS = ["LinearBaselineModel", "CpiPredConvModel", "CpiPredSelfAttnModel", "CpiPredCrossAttnModel"]
PREDICTION_TAGS = {"kcat", "KM", "Ki"}
RECOMPOSITION_TAGS = {"BasicRecomp", "AdvancedRecomp"}
SPLIT_TAGS = ["hpo_split"]

# Map model tags to the name used in the hydra config's architecture default
MODEL_TAG_TO_ARCH_NAME = {
    "LinearBaselineModel": "linear",
    "CpiPredConvModel": "cpi_pred_conv",
    "CpiPredSelfAttnModel": "cpi_pred_self_attn",
    "CpiPredCrossAttnModel": "cpi_pred_cross_attn",
}

def categorize_run(run: wandb.apis.public.Run) -> Optional[Dict[str, Any]]:
    """
    Analyzes a run's tags and summary to categorize it and determine its validity.
    
    Returns:
        A dictionary with categorization info, or None if the run is invalid/uncategorized.
    """
    run_tags = set(run.tags)
    
    # 1. Determine Task Type and Target
    found_pred_tags = run_tags.intersection(PREDICTION_TAGS)
    
    run_type = None
    target_param = None
    metric_to_sort_by = None

    if len(found_pred_tags) == 1:
        run_type = "single_task"
        target_param = found_pred_tags.pop()
        metric_to_sort_by = f"valid/{target_param}_Pearson"
    elif len(found_pred_tags) == 3:
        run_type = "multi_task"
        target_param = "all"
        # For multi-task, we must define a consistent metric to rank HPO runs.
        # Let's default to Ki_Pearson as in the original script.
        metric_to_sort_by = "valid/Ki_Pearson"
    else:
        # Skip runs that are not clearly single or multi-task
        return None

    # 2. Determine Recomposition Type
    if "BasicRecomp" in run_tags:
        recomposition = "basic"
    elif "AdvancedRecomp" in run_tags:
        recomposition = "advanced"
    else:
        recomposition = "none"

    # 3. Validate the Metric
    metric_value = run.summary.get(metric_to_sort_by)
    if metric_value is None or not isinstance(metric_value, (float, int)) or math.isnan(metric_value):
        # Skip if metric is missing, not a number, or is NaN
        return None

    return {
        "run_type": run_type,
        "target_param": target_param,
        "recomposition": recomposition,
        "metric_name": metric_to_sort_by,
        "metric_value": metric_value,
        "config": {k: v for k, v in run.config.items() if not k.startswith('_')}
    }


def main():
    """
    Main function to generate the master list of Hydra overrides for CV runs.
    """
    api = wandb.Api()
    master_run_list = []

    print("--- Generating Cross-Validation Run Definitions ---")
    
    for model_tag in MODEL_TAGS:
        arch_name = MODEL_TAG_TO_ARCH_NAME.get(model_tag)
        if not arch_name:
            print(f"  --> SKIPPING: No architecture mapping for model tag '{model_tag}'")
            continue

        # This dictionary will group all valid, categorized runs for the current model
        # Key: e.g., "single_task-kcat-none", "multi_task-all-basic"
        # Value: List of tuples [(metric_value, config), ...]
        grouped_runs = defaultdict(list)
        
        # 1. Query Broadly: Get all HPO runs for this model architecture
        query_tags = [model_tag] + SPLIT_TAGS
        filters = {"$and": [{"tags": tag} for tag in query_tags]}
        print(f"\nQuerying project '{PROJECT}' for model: '{model_tag}'...")
        
        try:
            runs = api.runs(f"{ENTITY}/{PROJECT}", filters=filters)
        except Exception as e:
            print(f"  --> An error occurred: {e}. Skipping.")
            continue

        # 2. Categorize and Validate Each Run
        for run in runs:
            run_info = categorize_run(run)
            if run_info:
                # Create a unique key for this specific configuration type
                category_key = (
                    f"{run_info['run_type']}-{run_info['target_param']}-{run_info['recomposition']}"
                )
                grouped_runs[category_key].append(
                    (run_info['metric_value'], run_info['config'])
                )
        
        if not grouped_runs:
            print(f"  --> No valid, completed HPO runs found for model '{model_tag}'.")
            continue

        # 3. Sort, Select Top K, and Generate Overrides for Each Group
        for category_key, runs_data in grouped_runs.items():
            print(f"  Processing category: {category_key} ({len(runs_data)} runs found)")
            
            # Sort by the metric value in descending order (higher is better)
            runs_data.sort(key=lambda x: x[0], reverse=True)
            
            top_k_configs = [config for metric, config in runs_data[:K_TOP_CONFIGS]]

            # Parse the category key back into structured info
            run_type, target, recomp = category_key.split('-')

            for i, config in enumerate(top_k_configs):
                for fold_index in range(5):
                    # Define target_columns override based on task type
                    if run_type == 'single_task':
                        target_columns_override = f"[{target}]"
                    else: # multi_task
                        target_columns_override = "[kcat,KM,Ki]"
                        
                    # Define recomposition override
                    # 'null' is how you override a value to be None in Hydra
                    recomp_override = recomp if recomp != "none" else "null"
                    
                    # Create the final dictionary of overrides
                    run_overrides = {
                        "architecture": arch_name,
                        "target_columns": target_columns_override,
                        "recomposition": recomp_override,
                        "data.fold_index": fold_index,
                        # Dynamically assign hyperparameter overrides to their correct location
                        **{f"model.lightning_module.{k}": v for k,v in config.items() if k in ['lr', 'optimizer']},
                        **{f"model.architecture.params.{k}": v for k,v in config.items() if k not in ['lr', 'optimizer', 'batch_size']},
                        **{f"data.{k}": v for k,v in config.items() if k in ['batch_size']}
                    }
                    master_run_list.append(run_overrides)

    print(f"\nGenerated a total of {len(master_run_list)} run configurations.")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_run_list, f, indent=2)
    print(f"Master run override list saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()