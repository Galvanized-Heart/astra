import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import re
import json

from astra.model.models.linear_baseline import LinearBaselineModel
from astra.data_processing.datamodules import AstraDataModule
from astra.model.modules.log10_stable_recomposition import *

# --- 1. FEATURE EXTRACTOR ---
class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_index: int):
        self.features = None
        # Hook into the penultimate Linear layer.
        # LinearBaselineModel.linear is a Sequential: [Linear, ReLU, Linear, ReLU, Linear]
        # Index 2 is the second Linear layer (dim_1 -> dim_2)
        model.linear[layer_index].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu().numpy()

# --- 2. MAIN SCRIPT ---
def run_ablation_inference(
    entity: str, 
    project: str, 
    tags: list, 
    ablation_mode: str = "full", 
    output_filename: str = "oof_predictions.pkl"
):
    api = wandb.Api()
    filters = {"tags": {"$all": tags}, "state": "finished"}
    
    # wandb API returns an iterator, cast to list so we can sort/filter
    runs_raw = list(api.runs(f"{entity}/{project}", filters=filters))
    print(f"    Found {len(runs_raw)} total runs for tags: {tags}")

    # --- Deduplication Logic ---
    # Group by fold and keep only the newest run
    runs_by_fold = {}
    for run in runs_raw:
        fold = 'unknown'
        for t in run.tags:
            match = re.search(r"fold[_-]?(\d+)", t)
            if match: 
                fold = match.group(1)
                break
        
        if fold != 'unknown':
            # If we haven't seen this fold, or if this run is newer than the one we have
            if fold not in runs_by_fold:
                runs_by_fold[fold] = run
            elif run.created_at > runs_by_fold[fold].created_at:
                runs_by_fold[fold] = run

    runs = list(runs_by_fold.values())
    print(f"    Filtered down to {len(runs)} unique latest runs for folds: {list(runs_by_fold.keys())}")
    # ---------------------------

    all_fold_data = []

    for run in runs:
        config = run.config
        
        # 1. Safely fish out dim_1 and dim_2 regardless of W&B nesting
        def get_nested(d, key, default):
            if key in d: return d[key]
            for k, v in d.items():
                if isinstance(v, dict):
                    res = get_nested(v, key, None)
                    if res is not None: return res
            return default
            
        dim_1 = get_nested(config, 'dim_1', 512)
        dim_2 = get_nested(config, 'dim_2', 128)
        
        # 2. Safely find target_columns
        target_columns = get_nested(config, 'target_columns', ['kcat', 'KM', 'Ki'])
        is_single_task = len(target_columns) == 1
        
        # 3. STRICTLY enforce out_dim based on the task (bypassing W&B config)
        if "AdvancedRecomp" in tags:
            recomp_func = elemtary_to_michaelis_menten_advanced_logspace
            out_dim = 5
        elif "BasicRecomp" in tags:
            recomp_func = elemtary_to_michaelis_menten_basic_logspace
            out_dim = 3
        else:
            recomp_func = None
            out_dim = 1 if is_single_task else 3

        # Get the fold string back for logging
        fold = 'unknown'
        for t in run.tags:
            match = re.search(r"fold[_-]?(\d+)", t)
            if match: fold = match.group(1); break
            
        print(f"        Processing Fold {fold} | Ablation: {ablation_mode}")

        ckpt_path = run.summary.get("last_local_checkpoint_path")
        if not ckpt_path or not Path(ckpt_path).exists():
            print(f"            -> WARNING: Checkpoint not found at {ckpt_path}")
            continue

        model = LinearBaselineModel(
            protein_emb_dim={'embedding': [None, 1280]}, 
            ligand_emb_dim={'embedding': [2048]}, 
            dim_1=dim_1, dim_2=dim_2, out_dim=out_dim
        )
        
        # FIXED: weights_only=False bypasses PyTorch 2.6 security restriction
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict']
        clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(clean_state_dict)
        model.eval().cuda()

        extractor = FeatureExtractor(model, layer_index=2)

        data_cfg_str = config.get('data_cfg', '')
        train_path = re.search(r"train_path=PosixPath\('([^']+)'\)", data_cfg_str).group(1)
        valid_path = re.search(r"valid_path=PosixPath\('([^']+)'\)", data_cfg_str).group(1)
        
        datamodule = AstraDataModule(
            train_path=train_path, 
            valid_path=valid_path, 
            batch_size=256, 
            target_columns=target_columns
        )
        datamodule.setup(stage="fit")
        val_loader = datamodule.val_dataloader()

        fold_results = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"            Inferencing"):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                targets = batch.pop("targets").cpu().numpy() 
                batch_size = targets.shape[0]

                raw_protein = batch['protein_embedding'].mean(dim=1).cpu().numpy() 
                raw_ligand = batch['ligand_embedding'].cpu().numpy()

                if ablation_mode == "protein_only":
                    batch['ligand_embedding'] = torch.zeros_like(batch['ligand_embedding'])
                elif ablation_mode == "ligand_only":
                    batch['protein_embedding'] = torch.zeros_like(batch['protein_embedding'])

                raw_output = model(**batch)
                embeddings = extractor.features
                
                rates_np = np.full((batch_size, 5), np.nan)
                preds_np = np.full((batch_size, 3), np.nan)
                true_np  = np.full((batch_size, 3), np.nan)
                
                param_to_idx = {'kcat': 0, 'KM': 1, 'Ki': 2}
                for i, param_name in enumerate(target_columns):
                    idx = param_to_idx[param_name]
                    true_np[:, idx] = targets[:, i] if is_single_task else targets[:, idx]

                if recomp_func:
                    rates_np[:, :raw_output.shape[1]] = raw_output.cpu().numpy()
                    preds_tensor = recomp_func(raw_output)
                    preds_np = preds_tensor.cpu().numpy()
                else:
                    preds_tensor = raw_output
                    for i, param_name in enumerate(target_columns):
                        idx = param_to_idx[param_name]
                        preds_np[:, idx] = preds_tensor[:, i].cpu().numpy()

                for i in range(batch_size):
                    fold_results.append({
                        "fold": fold,
                        "ablation": ablation_mode,
                        "true_kcat": true_np[i, 0],
                        "true_KM": true_np[i, 1],
                        "true_Ki": true_np[i, 2],
                        "pred_kcat": preds_np[i, 0],
                        "pred_KM": preds_np[i, 1],
                        "pred_Ki": preds_np[i, 2],
                        "log_k1": rates_np[i, 0],
                        "log_k_minus_1": rates_np[i, 1],
                        "log_k2": rates_np[i, 2],
                        "log_k_minus_2": rates_np[i, 3],
                        "log_k3": rates_np[i, 4],
                        "learned_embedding": embeddings[i],
                        "raw_protein_emb": raw_protein[i],
                        "raw_ligand_emb": raw_ligand[i]
                    })

        all_fold_data.extend(fold_results)

    df = pd.DataFrame(all_fold_data)
    df.to_pickle(output_filename)
    print(f"    Saved inference results to {output_filename}")

if __name__ == "__main__":
    entity = "lmse-university-of-toronto" # From your UI output
    project = "astra"
    
    # Run this script 3 times for the three states!
    # Adjust tags to match the exact run group you want to analyze
    tags = ["LinearBaselineModel", "AdvancedRecomp", "5fcv"] 

    print(f"Running ablation for {entity}/{project} with tags: {tags}")
    
    run_ablation_inference(entity, project, tags, ablation_mode="full", output_filename="results_full.pkl")
    run_ablation_inference(entity, project, tags, ablation_mode="protein_only", output_filename="results_prot_only.pkl")
    run_ablation_inference(entity, project, tags, ablation_mode="ligand_only", output_filename="results_ligand_only.pkl")