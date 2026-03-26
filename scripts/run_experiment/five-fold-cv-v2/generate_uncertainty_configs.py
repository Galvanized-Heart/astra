"""
This script probes W&B for the completed 5-fold CV runs.
It parses the wandb.group names to categorize runs into Direct, Basic, and Advanced.
It calculates the 5-fold average Pearson r to find the Champion for each category.
Finally, it generates 30 configs to test the Uncertainty Weighting strategy.
"""

import wandb
import json
import math
import os
import pandas as pd
from typing import Dict, Any, Optional
from collections import defaultdict

# --- CONFIGURATION ---
ENTITY = "lmse-university-of-toronto"
PROJECT = "astra"  

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "uncertainty_cv_configs.json")

# We only need to find champions for these two architectures
TARGET_MODELS = {
    "LinearBaselineModel": "linear",
    "CpiPredSelfAttnModel": "cpi_pred_self_attn"
}

# The target experiment modes we are generating for Hydra
TARGET_MODES = ["multi_task/direct", "multi_task/basic", "multi_task/advanced"]
FOLDS = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

KNOWN_HYPERPARAMETERS = {
    'lr': 'model.lightning_module.lr', 
    'batch_size': 'data.batch_size',
    # Linear Params
    'dim_1': 'architecture.params.dim_1',
    'dim_2': 'architecture.params.dim_2',
    # Attention Params
    'n_heads': 'architecture.params.n_heads',
    'd_ff': 'architecture.params.d_ff',
    'd_k': 'architecture.params.d_k',
    'd_v': 'architecture.params.d_v',
    'attn_out_dim': 'architecture.params.attn_out_dim',
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

def get_all_champions_for_architecture(api, model_tag):
    """
    Pulls runs for an architecture, categorizes them by mode by reading the 
    wandb.group string, and finds the 5-fold champion for each mode.
    """
    print(f"\n========================================================")
    print(f"--- Mining Champions for {model_tag} ---")
    print(f"========================================================")
    
    # Broad filter: Just get finished runs for this model
    filters = {
        "$and": [
            {"tags": model_tag},
            {"state": "finished"}
        ]
    }
    
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=filters))
    if not runs:
        raise ValueError(f"No finished runs found for {model_tag}!")

    # Bins to hold runs by mode
    categorized_runs = {
        "multi_task/direct": [],
        "multi_task/basic": [],
        "multi_task/advanced": []
    }

    for run in runs:
        group_name = run.group
        # We only care about runs that were part of an ensemble (have a group)
        if not group_name: 
            continue
            
        # Your previous script used "-all-" for multi-task runs. Skip single task.
        if "-all-" not in group_name:
            continue

        # Map the group string recomp keyword to our target modes
        mode = None
        if "-none-" in group_name:
            mode = "multi_task/direct"
        elif "-basic-" in group_name:
            mode = "multi_task/basic"
        elif "-advanced-" in group_name:
            mode = "multi_task/advanced"
            
        if not mode:
            continue

        # Extract metrics
        kcat_r = run.summary.get("valid/kcat_Pearson")
        km_r = run.summary.get("valid/KM_Pearson")
        ki_r = run.summary.get("valid/Ki_Pearson")
        
        # Skip if incomplete
        if any(v is None or math.isnan(v) for v in [kcat_r, km_r, ki_r]):
            continue
            
        avg_r = (kcat_r + km_r + ki_r) / 3.0
        
        categorized_runs[mode].append({
            "group": group_name,
            "avg_r": avg_r,
            "kcat_r": kcat_r, 
            "config": {k: v for k, v in run.config.items() if not k.startswith('_')}
        })

    champions = {}

    # Find the champion for each mode
    for mode, run_data in categorized_runs.items():
        print(f"\nEvaluating mode: {mode}")
        if not run_data:
            print(f"  -> WARNING: No valid runs found for {mode}. Skipping.")
            continue

        df = pd.DataFrame(run_data)
        
        # Group by ensemble member and average across folds
        grouped = df.groupby("group").agg(
            mean_avg_r=('avg_r', 'mean'),
            mean_kcat_r=('kcat_r', 'mean'),
            count=('group', 'count')
        ).reset_index()
        
        # Warn if we didn't get exactly 5 folds
        for _, row in grouped.iterrows():
            if row['count'] != 5:
                print(f"  -> Note: Group {row['group']} only has {int(row['count'])} finished folds.")
                
        # Sort: Highest Avg Pearson first, tie-break with kcat
        grouped = grouped.sort_values(by=["mean_avg_r", "mean_kcat_r"], ascending=[False, False])
        
        best_group = grouped.iloc[0]
        print(f"  -> CHAMPION: {best_group['group']}")
        print(f"  -> 5-Fold Avg Pearson: {best_group['mean_avg_r']:.4f}")
        
        champion_config = df[df["group"] == best_group["group"]].iloc[0]["config"]
        champions[mode] = {
            "group_name": best_group["group"],
            "hparams": extract_hpo_overrides(champion_config)
        }
        
    return champions


def main():
    api = wandb.Api()
    all_generated_configs = []

    print("--- Starting Generation Process ---")

    for model_tag, arch_name in TARGET_MODELS.items():
        # Get a dictionary of champions: { "multi_task/direct": {hparams...}, ... }
        champions_dict = get_all_champions_for_architecture(api, model_tag)

        # Generate 5 folds for every mode that successfully found a champion
        for mode in TARGET_MODES:
            if mode not in champions_dict:
                continue # Skip if we didn't find a champion for this mode
                
            champion_data = champions_dict[mode]
            hparams = champion_data["hparams"]
            manual_group_name = champion_data["group_name"]
            
            mode_short_name = mode.split('/')[-1].capitalize()
            arch_short_name = "SelfAttn" if "self_attn" in arch_name else "Linear"

            for fold in FOLDS:
                run_overrides = {
                    "experiment_mode": mode,
                    "architecture": arch_name,
                    "data": fold,
                    "mtl_strategy": "uncertainty",
                    "wandb.group": f"Uncertainty-{mode_short_name}-{arch_short_name}",
                    "extra_tags": ["uncertainty_exp", "5fcv"], 
                    "trainer.epochs": 20 
                }
                
                # Apply the extracted champion parameters
                run_overrides.update(hparams)
                all_generated_configs.append(run_overrides)

    # Save to JSON
    print(f"\n========================================================")
    print(f"Generated exactly {len(all_generated_configs)} configurations.")
    print(f"========================================================")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_generated_configs, f, indent=2)
    print(f"Saved to '{OUTPUT_FILE}'. Ready for submit_cv_jobs.py!")

if __name__ == "__main__":
    main()