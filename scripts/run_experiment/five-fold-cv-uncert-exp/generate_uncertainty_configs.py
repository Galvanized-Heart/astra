"""
This script probes W&B for the best performing 5-fold CV runs (from the 240 manual runs),
identifies the 'Champion' hyperparameters for Linear and Self-Attn architectures,
and generates exactly 30 configs to test the Uncertainty Weighting strategy.
"""

import wandb
import json
import math
import os
import pandas as pd
from typing import Dict, Any, Optional

ENTITY = "lmse-university-of-toronto"
PROJECT = "astra"  # Ensure this matches exactly

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "uncertainty_cv_configs.json")

# We only need to find champions for these two architectures
TARGET_MODELS = {
    "LinearBaselineModel": "linear",
    "CpiPredSelfAttnModel": "cpi_pred_self_attn"
}

# The modes and folds we want to generate for the new experiment
TARGET_MODES = ["direct", "basic", "advanced"]
FOLDS = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

# Re-use your robust extraction logic
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

def get_champion_hparams(api, model_tag):
    """Finds the best 5-fold average config for a given architecture."""
    print(f"\n--- Finding Champion for {model_tag} ---")
    
    # We look specifically at the 'advanced' multi-task runs to find the most robust model
    # Adjust the tags if your 240 runs used different specific tags
    filters = {
        "$and": [
            {"tags": model_tag},
            {"tags": "multi_task"}, # Assuming this tag exists, or use config filters
            {"config.experiment_mode": "multi_task/advanced"} 
        ]
    }
    
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=filters))
    
    if not runs:
        print(f"Warning: No runs found for {model_tag} in advanced mode.")
        # Fallback: remove experiment_mode filter and try again
        runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": model_tag}))
        if not runs:
            raise ValueError(f"No runs found at all for {model_tag}!")

    run_data = []
    for run in runs:
        # Get metrics
        kcat_r = run.summary.get("valid/kcat_Pearson")
        km_r = run.summary.get("valid/KM_Pearson")
        ki_r = run.summary.get("valid/Ki_Pearson")
        
        # Skip if run crashed or hasn't finished
        if any(v is None or math.isnan(v) for v in [kcat_r, km_r, ki_r]):
            continue
            
        avg_r = (kcat_r + km_r + ki_r) / 3.0
        group_name = run.group  # e.g., "LinearBaselineModel-all-advanced-valid/kcat_Pearson-top0"
        
        if not group_name:
            continue
            
        run_data.append({
            "group": group_name,
            "avg_r": avg_r,
            "kcat_r": kcat_r, # Keep for tie-breaking
            "config": {k: v for k, v in run.config.items() if not k.startswith('_')}
        })

    if not run_data:
        raise ValueError(f"No valid completed runs found for {model_tag}.")

    # Convert to DataFrame to easily average across the 5 folds
    df = pd.DataFrame(run_data)
    
    # Group by the ensemble member (wandb.group) and average the scores
    grouped = df.groupby("group").agg(
        mean_avg_r=('avg_r', 'mean'),
        mean_kcat_r=('kcat_r', 'mean'),
        count=('group', 'count')
    ).reset_index()
    
    # Sort by average R, tie-break with kcat R
    grouped = grouped.sort_values(by=["mean_avg_r", "mean_kcat_r"], ascending=[False, False])
    
    best_group = grouped.iloc[0]
    print(f"Champion found: {best_group['group']}")
    print(f"  -> 5-Fold Avg Pearson: {best_group['mean_avg_r']:.4f} (Count: {best_group['count']})")
    
    # Extract the actual hyperparameters from one of the runs in this winning group
    champion_config = df[df["group"] == best_group["group"]].iloc[0]["config"]
    overrides = extract_hpo_overrides(champion_config)
    
    return overrides

def main():
    api = wandb.Api()
    all_generated_configs = []

    print("--- Generating 30 Uncertainty Run Configurations ---")

    for model_tag, arch_name in TARGET_MODELS.items():
        # 1. Find the champion parameters for this architecture
        champion_hparams = get_champion_hparams(api, model_tag)
        print(f"  -> Extracted Params: {champion_hparams}")

        # 2. Apply these parameters across all 3 modes and 5 folds
        for mode in TARGET_MODES:
            for fold in FOLDS:
                run_overrides = {
                    "experiment_mode": f"multi_task/{mode}",
                    "architecture": arch_name,
                    "data": fold,
                    "model.lightning_module.mtl_strategy": "uncertainty",
                    "wandb.group": f"Uncertainty-{mode.capitalize()}-{arch_name.capitalize()}",
                    "trainer.epochs": 20
                }
                
                # Merge in the champion hyperparameters
                run_overrides.update(champion_hparams)
                all_generated_configs.append(run_overrides)

    # 3. Save to JSON
    print(f"\nGenerated exactly {len(all_generated_configs)} configurations.")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_generated_configs, f, indent=2)
    print(f"Saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()