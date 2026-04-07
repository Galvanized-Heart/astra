import hydra
from omegaconf import DictConfig
import wandb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from astra.model.lightning_models import AstraModule
from astra.data_processing.datamodules import AstraDataModule
from astra.model.modules.log10_stable_recomposition import *

@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    api = wandb.Api()
    
    all_fold_data = []

    for fold_idx, run_id in enumerate(cfg.run_ids):
        print(f"Processing Fold {fold_idx + 1} (Run ID: {run_id})...")
        
        # 1. Download best checkpoint from W&B
        run = api.run(f"{cfg.wandb_project}/{run_id}")
        artifacts = run.logged_artifacts()
        ckpt_artifact = [a for a in artifacts if a.type == 'model' and 'best' in a.aliases][0]
        ckpt_dir = ckpt_artifact.download()
        ckpt_path = f"{ckpt_dir}/model.ckpt"

        # 2. Load the model
        model = AstraModule.load_from_checkpoint(ckpt_path)
        model.eval()
        model.cuda()

        # 3. Setup Data (You need to load the VALIDATION set for this specific fold)
        # Assuming your datamodule knows how to load fold_idx
        datamodule = AstraDataModule(fold=fold_idx, batch_size=cfg.batch_size)
        datamodule.setup(stage="test") 
        val_loader = datamodule.val_dataloader()

        # 4. Inference Loop
        fold_results = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Inferencing Fold {fold_idx + 1}"):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                targets = batch.pop("targets").cpu().numpy() # True values
                
                raw_protein = batch['protein_embedding'].mean(dim=1).cpu().numpy() 
                raw_ligand = batch['ligand_embedding'].cpu().numpy()

                # Get the raw output from the base architecture
                raw_output = model.model(**batch)
                
                # Setup default NaNs for elementary rates (since Direct/Single models don't have them)
                batch_size = targets.shape[0]
                rates_np = np.full((batch_size, 5), np.nan) # [k1, k-1, k2, k-2, k3]
                preds_np = np.full((batch_size, 3), np.nan) # [kcat, KM, Ki]
                
                # --- BRANCH BASED ON RECOMPOSITION TYPE ---
                if cfg.recomposition_type in ["basic", "advanced"]:
                    # Model outputs elementary rates
                    rates_tensor = raw_output
                    rates_np[:, :rates_tensor.shape[1]] = rates_tensor.cpu().numpy()
                    
                    if cfg.recomposition_type == "basic":
                        preds_tensor = elemtary_to_michaelis_menten_basic_logspace(rates_tensor)
                    else:
                        preds_tensor = elemtary_to_michaelis_menten_advanced_logspace(rates_tensor)
                    
                    preds_np = preds_tensor.cpu().numpy()
                    
                else:
                    # Single-Task or Multi-Task Direct (No elementary rates)
                    preds_tensor = raw_output
                    
                    # Map the outputs to the correct columns based on model.target_columns
                    # e.g., if model.target_columns is ['kcat'], it goes into index 0
                    param_to_idx = {'kcat': 0, 'KM': 1, 'Ki': 2}
                    for i, param_name in enumerate(model.target_columns):
                        idx = param_to_idx[param_name]
                        preds_np[:, idx] = preds_tensor[:, i].cpu().numpy()

                # Store row-by-row
                for i in range(batch_size):
                    fold_results.append({
                        "fold": fold_idx,
                        "true_kcat": targets[i, 0],  # Assuming targets always has 3 cols (NaN padded)
                        "true_KM": targets[i, 1],
                        "true_Ki": targets[i, 2],
                        "pred_kcat": preds_np[i, 0],
                        "pred_KM": preds_np[i, 1],
                        "pred_Ki": preds_np[i, 2],
                        "log_k1": rates_np[i, 0],
                        "log_k_minus_1": rates_np[i, 1],
                        "log_k2": rates_np[i, 2],
                        "log_k_minus_2": rates_np[i, 3],
                        "log_k3": rates_np[i, 4],
                        "raw_protein_emb": raw_protein[i],
                        "raw_ligand_emb": raw_ligand[i]
                    })

        all_fold_data.extend(fold_results)

    # 5. Save everything to a massive DataFrame
    df = pd.DataFrame(all_fold_data)
    
    # Calculate Errors here so they are ready for UMAP coloring
    df['error_KM'] = abs(df['true_KM'] - df['pred_KM'])
    df['error_kcat'] = abs(df['true_kcat'] - df['pred_kcat'])
    
    df.to_pickle(cfg.output_csv.replace(".csv", ".pkl")) # Pickle saves numpy arrays safely
    print(f"Saved inference results to {cfg.output_csv}")

if __name__ == "__main__":
    main()