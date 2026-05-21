import click
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from astra.pipelines.evaluation.fetcher import RunFetcher
from astra.pipelines.evaluation.builder import InferenceModelBuilder
from astra.pipelines.evaluation.recomposer import Recomposer
from astra.pipelines.evaluation.engine import InferenceEngine
from astra.constants import PROJECT_ROOT

@click.command()
@click.option('--entity', default="lmse-university-of-toronto", help='W&B Entity')
@click.option('--project', default="astra", help='W&B Project')
@click.option('--tags', multiple=True, required=True, help='W&B tags to filter runs (e.g. --tags 5fcv --tags LinearBaselineModel)')
@click.option('--out_dir', default="results/inference_data", help='Directory to save parquet files')
def main(entity, project, tags, out_dir):
    """
    Runs full inference and extracts embeddings for UMAP plotting.
    """
    output_dir = PROJECT_ROOT / out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Inference Pipeline on {device} ---")

    fetcher = RunFetcher(entity, project)
    runs = fetcher.get_runs(list(tags))
    
    if not runs:
        print("No runs found. Exiting.")
        return
        
    for run in runs:
        print(f"\n{'='*50}")
        print(f"Processing Run: {run.name}")
        print(f"Group: {run.group}")
        print(f"{'='*50}")
        
        # 1. Fetch Config and Checkpoint
        config = fetcher.get_run_config(run)
        ckpt_path = fetcher.get_checkpoint_path(run)
        
        if not ckpt_path or not ckpt_path.exists():
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping run.")
            continue
            
        # 2. Build and Load Model
        try:
            builder, model = InferenceModelBuilder.build(config, str(ckpt_path))
            model = model.to(device)
            builder.datamodule.setup(stage="validate")
            val_loader = builder.datamodule.val_dataloader()
        except Exception as e:
            print(f"⚠️ Failed to build model for {run.name}: {e}")
            continue

        # 3. Setup Recomposer & Engine
        recomp_func = config['model']['lightning_module'].get('recomposition_func')
        target_cols = config['data']['target_columns']
        arch_name = config['model']['architecture']['name']
        
        recomposer = Recomposer(recomp_func, target_cols)
        engine = InferenceEngine(model, arch_name, recomposer)
        
        # 4. Inference Loop
        all_data = []
        
        print("Running validation batches...")
        for batch in tqdm(val_loader, desc="Inference"):
            # Move batch to GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                results = engine.run_batch(batch)
                
            # Unbatch and format
            batch_size = results["predictions"].shape[0]
            for i in range(batch_size):
                all_data.append({
                    "run_id": run.id,
                    "group": run.group,
                    "fold": next((t for t in run.tags if 'fold' in t), "unknown"),
                    "true_kcat": results["true_targets"][i, 0],
                    "true_KM": results["true_targets"][i, 1],
                    "true_Ki": results["true_targets"][i, 2],
                    "pred_kcat": results["predictions"][i, 0],
                    "pred_KM": results["predictions"][i, 1],
                    "pred_Ki": results["predictions"][i, 2],
                    "raw_protein_emb": results["raw_protein_emb"][i],
                    "raw_ligand_emb": results["raw_ligand_emb"][i],
                    "learned_emb": results["learned_emb"][i]
                })

        # 5. Save to Parquet
        df = pd.DataFrame(all_data)
        
        # Clean the run name to be safe for filenames
        safe_name = run.name.replace("/", "_").replace("\\", "_")
        save_path = output_dir / f"{safe_name}_inference.parquet"
        
        df.to_parquet(save_path)
        print(f"✅ Saved {len(df)} rows to {save_path}")
        
        # Cleanup hook to free memory before next run
        engine.cleanup()
        torch.cuda.empty_cache()

    print("\n🎉 Pipeline Complete! All models processed.")

if __name__ == "__main__":
    main()