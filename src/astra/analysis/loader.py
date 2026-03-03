import wandb
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Any
import json

class RunLoader:
    def __init__(self, entity: str, project: str):
        self.api = wandb.Api()
        self.entity = entity
        self.project = project

    def get_runs(self, tag_filter: List[str]) -> List[wandb.apis.public.Run]:
        """Fetch runs matching all tags."""
        filters = {
            "tags": {"$all": tag_filter}, 
            "state": {"$in": ["finished", "crashed"]}
        }
        runs = self.api.runs(f"{self.entity}/{self.project}", filters=filters)
        print(f"Found {len(runs)} runs matching tags: {tag_filter}")
        return runs

    def _get_run_config_key(self, config: dict) -> str:
        """
        Creates a deterministic, hashable key from the run configuration by 
        excluding non-hyperparameter metadata and serializing the result.
        
        This key identifies the unique set of hyperparameters used for training 
        on a specific dataset split (fold).
        """
        
        EXCLUDE_KEYS = [
            '_wandb', 'hydra', 'group', 'job_type', 
            'wandb', '_timestamp', '_runtime', '_step', 
            'optimization_goal_metric' 
        ]
        
        # Filter the configuration
        filtered_config = {}
        for k, v in config.items():
            # Check if key should be excluded, or if value is metadata dict
            if k not in EXCLUDE_KEYS and not (isinstance(v, dict) and '_type' in v):
                if isinstance(v, dict):
                    v_cleaned = {
                        inner_k: inner_v 
                        for inner_k, inner_v in v.items() 
                        if not inner_k.startswith('_')
                    }
                    if v_cleaned:
                        filtered_config[k] = v_cleaned
                else:
                    filtered_config[k] = v

        # Serialize the filtered config deterministically
        try:
            config_key = json.dumps(filtered_config, sort_keys=True)
            return config_key
        except TypeError:
            return str(filtered_config)

    def _deduplicate_runs(self, runs: List[wandb.apis.public.Run]) -> List[wandb.apis.public.Run]:
        """
        Filters a list of runs, keeping only the first run encountered for 
        each unique configuration fingerprint.
        """
        unique_keys = set()
        deduplicated_runs = []
        
        print("Starting configuration deduplication...")

        for run in runs:
            # Generate the unique key based on hyperparams and data split
            key = self._get_run_config_key(run.config)
            
            # Check for duplication
            if key not in unique_keys:
                unique_keys.add(key)
                deduplicated_runs.append(run)
        return deduplicated_runs

    def _derive_metadata(self, config: dict) -> Tuple[str, str]:
        """
        Infers clean Architecture and Experiment Mode labels based on the 
        verified nested config structure.
        """
        # Get achitecture
        try:
            arch_raw = config['model']['architecture']['name']
        except (KeyError, TypeError):
            arch_raw = config.get('architecture.name', 'Unknown')

        # Map raw class names to pretty names
        arch_map = {
            "LinearBaselineModel": "Linear",
            "CpiPredConvModel": "Conv1D",
            "CpiPredSelfAttnModel": "Self-Attn",
            "CpiPredCrossAttnModel": "Cross-Attn"
        }
        arch = arch_map.get(arch_raw, arch_raw) # Returns raw name if not in map

        # Get experiment mode
        try:
            targets = config['data']['target_columns']
        except (KeyError, TypeError):
            targets = config.get('data.target_columns', [])

        # Path: model -> lightning_module -> recomposition_func
        try:
            recomp = config['model']['lightning_module']['recomposition_func']
        except (KeyError, TypeError):
            recomp = config.get('model.lightning_module.recomposition_func')

        # Logic to determine Mode
        is_single_task = False
        
        # Handle targets being a list OR a string representation of a list
        if isinstance(targets, list):
            is_single_task = (len(targets) == 1)
        elif isinstance(targets, str):
            # If string contains comma, it's likely multi-task e.g. "['kcat', 'KM']"
            # If no comma, likely single task e.g. "['kcat']" or "kcat"
            is_single_task = (',' not in targets)

        if is_single_task:
            mode = "Single Task"
        elif recomp == "BasicRecomp":
            mode = "Multi-Task (Basic)"
        elif recomp == "AdvancedRecomp":
            mode = "Multi-Task (Advanced)"
        else:
            # Multi-task implies 3 targets, if recomp is None/Null, it's Direct
            mode = "Multi-Task (Direct)"

        return arch, mode

    def load_predictions(self, runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
        """
        Aggregates prediction CSVs from W&B runs.
        """
        all_preds = []

        for run in tqdm(runs, desc="Loading Run Data"):
            # 1. Derive Metadata
            arch, mode = self._derive_metadata(run.config)
            
            # 2. Extract Fold from W&B Tags
            # Looks for "fold_X", "fold-X", or "hpo_split"
            fold = 'fold_unknown'
            for t in run.tags:
                # Check for standard fold tags
                match = re.search(r"fold[_-]?(\d+)", t)
                if match:
                    fold = f"{match.group(1)}"
                    break
                # Check for HPO split
                if t == 'hpo_split':
                    fold = 'hpo'
            
            # 3. Locate Prediction File
            # We check both best and last to ensure we get data
            pred_path_str = run.summary.get("best_predictions_path") # or run.summary.get("last_predictions_path")

            if not pred_path_str:
                continue

            pred_path = Path(pred_path_str)
            
            # Simple check if file exists (for running on cluster)
            if not pred_path.exists():
                continue

            try:
                df = pd.read_csv(pred_path)
                
                # Add Metadata Columns for Grouping
                df['run_id'] = run.id
                df['architecture'] = arch
                df['experiment_mode'] = mode
                df['fold'] = fold
                
                all_preds.append(df)
            except Exception:
                continue

        if not all_preds:
            return pd.DataFrame()
            
        return pd.concat(all_preds, ignore_index=True)