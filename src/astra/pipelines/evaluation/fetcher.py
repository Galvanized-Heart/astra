import wandb
from pathlib import Path
from typing import List, Dict, Any, Optional

class RunFetcher:
    def __init__(self, entity: str, project: str):
        self.api = wandb.Api()
        self.entity = entity
        self.project = project

    def get_runs(self, tags: List[str]) -> List[wandb.apis.public.Run]:
        """Fetch runs matching all provided tags."""
        filters = {
            "tags": {"$all": tags},
            "state": "finished"
        }
        runs = list(self.api.runs(f"{self.entity}/{self.project}", filters=filters))
        print(f"Found {len(runs)} finished runs matching tags: {tags}")
        return runs

    def unflatten_config(self, flat_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a W&B flattened config back into a nested Hydra dictionary.
        e.g., {'architecture.params.dim_1': 512} -> {'architecture': {'params': {'dim_1': 512}}}
        """
        nested_config = {}
        for key, value in flat_config.items():
            # Skip wandb internal keys
            if key.startswith('_'):
                continue
                
            parts = key.split('.')
            current_level = nested_config
            
            # Traverse/create nested dictionaries for all but the last key part
            for part in parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
                
            # Set the actual value at the deepest level
            current_level[parts[-1]] = value
            
        return nested_config

    def get_run_config(self, run: wandb.apis.public.Run) -> Dict[str, Any]:
        """Extracts and un-flattens the config from a run."""
        # W&B sometimes stores the raw config inside run.config
        raw_config = {k: v for k, v in run.config.items()}
        return self.unflatten_config(raw_config)

    def get_checkpoint_path(self, run: wandb.apis.public.Run, prefer: str = "last") -> Optional[Path]:
        """
        Extracts the local checkpoint path from the run's summary.
        """
        # We look for the keys you established in PipelineBuilder
        key = f"{prefer}_local_checkpoint_path"
        ckpt_str = run.summary.get(key)
        
        # Fallback if 'last' isn't found, try 'best'
        if not ckpt_str and prefer == "last":
            ckpt_str = run.summary.get("best_local_checkpoint_path")
            
        if not ckpt_str:
            return None
            
        return Path(ckpt_str)