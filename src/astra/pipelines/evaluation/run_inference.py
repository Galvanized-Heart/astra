import click
import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

from astra.pipelines.evaluation.fetcher import RunFetcher
from astra.pipelines.evaluation.builder import InferenceModelBuilder
from astra.pipelines.evaluation.recomposer import Recomposer
from astra.pipelines.evaluation.engine import InferenceEngine
from astra.constants import PROJECT_ROOT

@click.command()
@click.option('--manifest', default="results/inference_manifest.json", help='Path to JSON manifest')
@click.option('--out_dir', default="results/inference_data", help='Directory to save parquet files')
def main(manifest, out_dir):
    """Runs full inference and extracts embeddings for UMAP plotting."""
    manifest_path = PROJECT_ROOT / manifest
    if not manifest_path.exists():
        print(f"❌ Manifest not found at {manifest_path}")
        return
        
    with open(manifest_path, 'r') as f:
        runs_to_process = json.load(f)

    output_dir = PROJECT_ROOT / out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Inference Pipeline on {device} ---")

    fetcher = RunFetcher("lmse-university-of-toronto", "astra")
    
    for run_data in runs_to_process:
        print(f"\n{'='*50}\nProcessing Run: {run_data['run_name']}\nGroup: {run_data['group']} | Fold: {run_data['fold']}\n{'='*50}")
        
        config = fetcher.unflatten_config(run_data['config'])
        ckpt_path = Path(run_data['ckpt_path'])
        
        if not ckpt_path.exists():
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping run.")
            continue
            
        # 1. Load Original Metadata CSV
        # This allows us to map predictions back to the exact protein and ligand!
        valid_csv_path = PROJECT_ROOT / config['data']['valid_path']
        metadata_df = pd.read_csv(valid_csv_path)
        
        # 2. Build and Load Model
        try:
            builder, model = InferenceModelBuilder.build(config, str(ckpt_path))
            model = model.to(device)
            builder.datamodule.setup(stage="validate")
            val_loader = builder.datamodule.val_dataloader()
        except Exception as e:
            print(f"⚠️ Failed to build model for {run_data['run_name']}: {e}")
            continue

        # 3. Setup Recomposer & Engine
        recomp_func = config['model']['lightning_module'].get('recomposition_func')
        target_cols = config['data']['target_columns']
        recomposer = Recomposer(recomp_func, target_cols)
        engine = InferenceEngine(model, recomposer)
        
        # 4. Inference Loop
        all_data = []
        global_idx = 0 # Tracks our row in the metadata CSV
        
        print("Running validation batches...")
        for batch in tqdm(val_loader, desc="Inference"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                results = engine.run_batch(batch)
                
            batch_size = results["predictions"].shape[0]
            for i in range(batch_size):
                # Grab the matching metadata strings
                protein_seq = metadata_df.iloc[global_idx]['protein_sequence']
                ligand_smi = metadata_df.iloc[global_idx]['ligand_smiles']
                
                all_data.append({
                    "protein_sequence": protein_seq,
                    "ligand_smiles": ligand_smi,
                    "arch": run_data['arch_clean'],
                    "mode": run_data['mode_clean'],
                    "fold": run_data['fold'],
                    "true_kcat": results["true_targets"][i, 0],
                    "true_KM": results["true_targets"][i, 1],
                    "true_Ki": results["true_targets"][i, 2],
                    "pred_kcat": results["predictions"][i, 0],
                    "pred_KM": results["predictions"][i, 1],
                    "pred_Ki": results["predictions"][i, 2],
                    "raw_protein_emb": results["raw_protein_emb"][i],
                    "raw_ligand_emb": results["raw_ligand_emb"][i],
                })
                global_idx += 1

        # 5. Save to Parquet
        df = pd.DataFrame(all_data)
        safe_name = run_data['run_name'].replace("/", "_").replace("\\", "_")
        save_path = output_dir / f"{safe_name}_{run_data['fold']}_inference.parquet"
        
        df.to_parquet(save_path)
        print(f"✅ Saved {len(df)} rows to {save_path}")
        
        torch.cuda.empty_cache()

    print("\n🎉 Pipeline Complete! All models processed.")

if __name__ == "__main__":
    main()