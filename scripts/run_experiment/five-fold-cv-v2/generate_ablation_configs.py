import wandb
import json
import math
import os
import pandas as pd
from typing import Dict, Any, Optional

# --- CONFIGURATION ---
ENTITY = "lmse-university-of-toronto"
PROJECT = "astra"  

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "ablation_cv_configs.json")

TARGET_MODEL = "LinearBaselineModel"
TARGET_MODEL_HYDRA = "linear"
FOLDS = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
TARGETS = ["kcat", "KM", "Ki"]
ABLATION_MODES = ["protein_only", "ligand_only"]

KNOWN_HYPERPARAMETERS = {
    'lr': 'model.lightning_module.lr', 
    'batch_size': 'data.batch_size',
    'dim_1': 'architecture.params.dim_1',
    'dim_2': 'architecture.params.dim_2',
}

def find_value_in_nested_dict(key_to_find: str, nested_dict: Dict[str, Any]) -> Optional[Any]:
    if key_to_find in nested_dict:
        return nested_dict[key_to_find]
    for value in nested_dict.values():
        if isinstance(value, dict):
            if (found_value := find_value_in_nested_dict(key_to_find, value)) is not None:
                return found_value
    return None

def extract_hpo_overrides(wandb_config: Dict[str, Any]) -> Dict[str, Any]:
    overrides = {}
    for param_name, hydra_path in KNOWN_HYPERPARAMETERS.items():
        found_value = find_value_in_nested_dict(param_name, wandb_config)
        if found_value is None and hydra_path in wandb_config:
            found_value = wandb_config[hydra_path]
        if found_value is not None:
            overrides[hydra_path] = found_value
    return overrides

def get_single_task_champions(api):
    print(f"\n========================================================")
    print(f"--- Mining Single-Task Champions for Linear Model ---")
    print(f"========================================================")
    
    filters = {
        "$and": [
            {"tags": TARGET_MODEL},
            {"state": "finished"}
        ]
    }
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=filters))
    
    # Bins to hold runs by target
    categorized_runs = { "kcat": [], "KM": [], "Ki": [] }

    for run in runs:
        group_name = run.group
        if not group_name: continue
            
        # Identify Single Task runs (Assuming they don't contain "-all-")
        if "-all-" in group_name: continue

        # Identify which target this run was predicting
        target = None
        for t in TARGETS:
            # Check if target name is in the group name (e.g., "LinearBaselineModel-kcat-none")
            if f"-{t}-" in group_name or group_name.endswith(f"-{t}") or f"_{t}_" in group_name:
                target = t
                break
        
        # Fallback: check config
        if not target:
            target_cols = find_value_in_nested_dict('target_columns', run.config)
            if isinstance(target_cols, list) and len(target_cols) == 1:
                target = target_cols[0]
                
        if not target or target not in categorized_runs: continue

        # Extract metric
        metric_val = run.summary.get(f"valid/{target}_Pearson")
        if metric_val is None or math.isnan(metric_val): continue
        
        categorized_runs[target].append({
            "group": group_name,
            "metric": metric_val,
            "config": {k: v for k, v in run.config.items() if not k.startswith('_')}
        })

    champions = {}

    for target, run_data in categorized_runs.items():
        print(f"\nEvaluating Target: {target}")
        if not run_data:
            print(f"  -> WARNING: No valid runs found for {target}. Skipping.")
            continue

        df = pd.DataFrame(run_data)
        grouped = df.groupby("group").agg(
            mean_metric=('metric', 'mean'), count=('group', 'count')
        ).reset_index()
                
        grouped = grouped.sort_values(by="mean_metric", ascending=False)
        best_group = grouped.iloc[0]
        
        print(f"  -> CHAMPION: {best_group['group']} (Folds: {int(best_group['count'])})")
        print(f"  -> Avg Pearson: {best_group['mean_metric']:.4f}")
        
        champion_config = df[df["group"] == best_group["group"]].iloc[0]["config"]
        champions[target] = extract_hpo_overrides(champion_config)
        
    return champions

def main():
    api = wandb.Api()
    all_generated_configs = []

    champions_dict = get_single_task_champions(api)

    # Map the targets to your specific Hydra config files
    experiment_mode_map = {
        "kcat": "single_task/kcat_only",
        "KM": "single_task/km_only",
        "Ki": "single_task/ki_only"
    }

    for target, hparams in champions_dict.items():
        for mode in ABLATION_MODES:
            mode_formatted = "ProteinOnly" if mode == "protein_only" else "LigandOnly"
            
            for fold in FOLDS:
                run_overrides = {
                    "experiment_mode": experiment_mode_map[target], # <--- UPDATED HERE
                    "architecture": TARGET_MODEL_HYDRA,
                    "data": fold,
                    # "data.target_columns" override removed, since the YAML handles it!
                    "+model.lightning_module.ablation_mode": mode, 
                    "wandb.group": f"Ablation-{mode_formatted}-Linear-{target}",
                    "extra_tags": ["ablation_exp", "5fcv"], 
                    "trainer.epochs": 20 
                }
                run_overrides.update(hparams)
                all_generated_configs.append(run_overrides)

    print(f"\n========================================================")
    print(f"Generated exactly {len(all_generated_configs)} configurations.")
    print(f"========================================================")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_generated_configs, f, indent=2)
    print(f"Saved to '{OUTPUT_FILE}'. Ready for submission!")

if __name__ == "__main__":
    main()