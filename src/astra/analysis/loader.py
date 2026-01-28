import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

class RunLoader:
    def __init__(self, entity: str, project: str):
        self.api = wandb.Api()
        self.entity = entity
        self.project = project

    def get_runs(self, tag_filter: List[str]) -> List[wandb.apis.public.Run]:
        """Fetch runs matching all tags."""
        filters = {"tags": {"$all": tag_filter}, "state": "finished"}
        runs = self.api.runs(f"{self.entity}/{self.project}", filters=filters)
        print(f"Found {len(runs)} runs matching tags: {tag_filter}")
        return runs

    def load_predictions(self, runs: List[wandb.apis.public.Run], download_root: Path = None) -> pd.DataFrame:
        """
        Aggregates prediction CSVs. 
        If running on the cluster, it reads the path logged in summary.
        """
        all_preds = []

        for run in tqdm(runs, desc="Loading Run Data"):
            # Extract config/metadata
            arch = run.config.get('architecture', {}).get('name', 'Unknown')
            mode = run.config.get('experiment_mode', 'Unknown')
            # Extract fold from tags (assuming format 'fold_X')
            fold = next((t for t in run.tags if t.startswith('fold_')), 'fold_unknown')
            
            # Locate prediction file
            # Note: This relies on PredictionSaver logging 'best_predictions_path' to summary
            pred_path_str = run.summary.get("best_predictions_path")
            
            if not pred_path_str:
                print(f"Warning: No prediction path found for run {run.name}")
                continue

            pred_path = Path(pred_path_str)
            
            # Logic if running locally and files are on cluster:
            # You might need to implement W&B Artifact download here if you change PredictionSaver later.
            if not pred_path.exists():
                print(f"Warning: File {pred_path} not found locally. Skipping.")
                continue

            try:
                df = pd.read_csv(pred_path)
                df['run_id'] = run.id
                df['architecture'] = arch
                df['experiment_mode'] = mode
                df['fold'] = fold
                all_preds.append(df)
            except Exception as e:
                print(f"Error reading {pred_path}: {e}")

        if not all_preds:
            return pd.DataFrame()
            
        return pd.concat(all_preds, ignore_index=True)