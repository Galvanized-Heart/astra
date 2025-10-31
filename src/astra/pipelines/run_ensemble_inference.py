import pandas as pd
import torch
import lightning as L
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from astra.model.lightning_models import AstraModule
from astra.data_processing.datamodules import AstraDataModule
from astra.data_processing.featurizers import ESMFeaturizer, MorganFeaturizer
from astra.model.modules.kinetics import elemtary_to_michaelis_menten_basic, elemtary_to_michaelis_menten_advanced
from torchmetrics.collections import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, PearsonCorrCoef, SpearmanCorrCoef, KendallRankCorrCoef



def select_top_models(results_csv_path: str, n_top: int = 5):
    """
    Parses the results CSV to find the top N models for each experimental group.

    Args:
        results_csv_path (str): Path to your CSV file containing model results.
        n_top (int): The number of top models to select for each group.

    Returns:
        dict: A dictionary where keys are group identifiers (tuples) and
              values are lists of the top N checkpoint paths.
    """
    df = pd.read_csv(results_csv_path)

    # --- Data Cleaning & Preprocessing ---
    # Drop runs that failed or didn't save a checkpoint
    df.dropna(subset=['best_local_checkpoint_path'], inplace=True)
    df = df[df['best_local_checkpoint_path'] != '']

    # --- Identify Model Type and Grouping Keys ---
    # We create a unique key for each experimental setup
    def get_group_key(row):
        tags = row['Tags'].split(', ')
        
        # Find fold
        fold = next((tag for tag in tags if 'fold_' in tag), None)
        
        # Determine task type (multi_task vs single_task)
        task_type = 'multi_task' if 'multi_task' in tags else 'single_task'
        
        # For single-task, identify the target
        target = 'all'
        if task_type == 'single_task':
            if 'target_KCAT' in tags: target = 'kcat'
            elif 'target_KM' in tags: target = 'KM'
            elif 'target_KI' in tags: target = 'Ki'

        # Determine recomposition function
        recomp = row.get('model.lightning_module.recomposition_func', 'direct')
        if pd.isna(recomp) or recomp == '':
            recomp = 'direct'

        return (fold, task_type, target, recomp)

    df['group_key'] = df.apply(get_group_key, axis=1)

    # --- Sort and Select Top N ---
    # For multi-task, we sort by validation loss. For single-task, by the primary metric.
    # This is a sensible default; you can adjust the sort_by logic.
    def get_sort_metric(row):
        if row['group_key'][1] == 'multi_task':
            # For multi-task models, overall validation loss is a good proxy
            return 'valid_loss_epoch' # Assuming you log this
        else:
            # For single-task, sort by the primary correlation metric for that task
            target_map = {'kcat': 'kcat', 'KM': 'KM', 'Ki': 'Ki'}
            target_key = target_map[row['group_key'][2]]
            # Let's use Pearson, but you could choose Spearman
            return f'valid/{target_key}_Pearson'

    df['sort_by_metric_name'] = df.apply(get_sort_metric, axis=1)
    
    # We need to handle metrics where higher is better (correlations) vs. lower is better (loss)
    # This is a bit tricky since the metric name changes per row. A simple way:
    all_checkpoints = defaultdict(list)
    for group_key, group_df in df.groupby('group_key'):
        metric_name = group_df['sort_by_metric_name'].iloc[0]
        ascending = 'loss' in metric_name # Sort ascending if it's a loss
        
        # Sort the group and take the top N
        top_n_df = group_df.sort_values(by=metric_name, ascending=ascending).head(n_top)
        
        all_checkpoints[group_key] = top_n_df['best_local_checkpoint_path'].tolist()

    return dict(all_checkpoints)



# --- Helper function to generate predictions ---
def get_predictions(model: AstraModule, dataloader: torch.utils.data.DataLoader, device: str):
    """Generates predictions for a given model and dataloader."""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            targets = batch.pop("targets")
            
            output = model.model(**batch)
            
            # Note: We are getting the RAW model output here, before recomposition
            # Recomposition will be applied *after* averaging.
            all_preds.append(output.cpu())
            all_targets.append(targets.cpu())
            
    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)



# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    RESULTS_CSV = Path("path/to/your/results.csv")
    # You need a base config for featurizers and data paths.
    # We will overwrite the train/valid paths for each fold.
    BASE_CONFIG_FOR_DATA = {
        'protein_featurizer': {'name': 'ESMFeaturizer', 'params': {}}, # Or whatever you used
        'ligand_featurizer': {'name': 'MorganFeaturizer', 'params': {}},
        'batch_size': 64, # Use a larger batch size for inference
        'featurizer_batch_size': 64,
        'target_columns': ['kcat', 'KM', 'Ki'],
        'target_transform': None # Set to what you used in training
    }
    # Assume your validation data is named like this:
    VALID_DATA_PATH_TEMPLATE = "data/davis_కినాਸੇ/folds/davis_test_fold_{fold_num}.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_TOP_MODELS = 5
    
    # --- 1. Select Top 5 Models for Each Group ---
    print("Selecting top models for ensembling...")
    ensemble_groups = select_top_models(RESULTS_CSV, n_top=N_TOP_MODELS)
    print(f"Found {len(ensemble_groups)} groups to ensemble.")

    # --- 2. Initialize Metrics and Results Storage ---
    target_columns = BASE_CONFIG_FOR_DATA['target_columns']
    metric_collections = {}
    for param_name in target_columns:
        metric_collections[param_name] = MetricCollection({
            'Pearson': PearsonCorrCoef(),
            'Spearman': SpearmanCorrCoef(),
            'R2': R2Score(),
            'Kendall': KendallRankCorrCoef(),
            'MSE': MeanSquaredError(),
            'RMSE': MeanSquaredError(squared=False),
            'MAE': MeanAbsoluteError()
        }).to(DEVICE)
    
    all_results = []
    
    # --- 3. Loop Through Groups, Predict, Ensemble, and Evaluate ---
    for group_key, checkpoint_paths in ensemble_groups.items():
        fold, task_type, target, recomp = group_key
        print(f"\n--- Processing Group: {group_key} ---")
        
        if len(checkpoint_paths) < N_TOP_MODELS:
            print(f"WARNING: Found only {len(checkpoint_paths)} models for this group. Skipping.")
            continue
            
        # --- A. Setup DataModule for the current fold ---
        fold_num = fold.split('_')[-1]
        valid_path = VALID_DATA_PATH_TEMPLATE.format(fold_num=fold_num)
        
        # Instantiate featurizers (as in your builder)
        protein_featurizer = ESMFeaturizer(**BASE_CONFIG_FOR_DATA['protein_featurizer']['params'], device=DEVICE)
        ligand_featurizer = MorganFeaturizer(**BASE_CONFIG_FOR_DATA['ligand_featurizer']['params'])
        
        dm = AstraDataModule(
            data_paths={'valid': valid_path},
            protein_featurizer=protein_featurizer,
            ligand_featurizer=ligand_featurizer,
            **BASE_CONFIG_FOR_DATA
        )
        dm.setup('validate')
        dataloader = dm.val_dataloader()

        # --- B. Generate Predictions from all models in the group ---
        model_predictions = []
        true_targets = None
        for ckpt_path in checkpoint_paths:
            print(f"  Loading model from {Path(ckpt_path).name}")
            model = AstraModule.load_from_checkpoint(ckpt_path, map_location=DEVICE)
            preds, targets = get_predictions(model, dataloader, DEVICE)
            model_predictions.append(preds)
            if true_targets is None:
                true_targets = targets

        # --- C. Ensemble the predictions ---
        # Stack predictions: shape becomes (N_MODELS, N_SAMPLES, N_OUTPUTS)
        predictions_stack = torch.stack(model_predictions, dim=0)
        # Average across models: shape becomes (N_SAMPLES, N_OUTPUTS)
        avg_raw_preds = torch.mean(predictions_stack, dim=0)

        # --- D. Apply Recomposition or Handle Single-Task ---
        if task_type == 'multi_task':
            if recomp == 'BasicRecomp':
                final_preds = elemtary_to_michaelis_menten_basic(avg_raw_preds)
            elif recomp == 'AdvancedRecomp':
                final_preds = elemtary_to_michaelis_menten_advanced(avg_raw_preds)
            else: # 'direct'
                final_preds = avg_raw_preds
        
        elif task_type == 'single_task':
            # This is the special case. We need to combine predictions from 3 different groups.
            # We will process them later after all single-task predictions are generated.
            # For now, just store the single-task predictions.
            if 'single_task_preds' not in locals():
                single_task_preds = defaultdict(dict)
            single_task_preds[fold][target] = avg_raw_preds
            continue # Skip metric calculation for now
            
        # --- E. Calculate Metrics for the ensemble ---
        group_metrics = {'fold': fold, 'task_type': task_type, 'recomp': recomp}
        for i, param_name in enumerate(target_columns):
            preds_i = final_preds[:, i].to(DEVICE)
            targets_i = true_targets[:, i].to(DEVICE)
            
            mask = ~torch.isnan(targets_i)
            if mask.sum() > 0:
                metrics = metric_collections[param_name](preds_i[mask], targets_i[mask])
                # Add prefix for clarity in final table
                metrics_renamed = {f"valid/{param_name}_{k}": v.item() for k, v in metrics.items()}
                group_metrics.update(metrics_renamed)

        all_results.append(group_metrics)

    # --- 4. Process Single-Task Ensembles ---
    if 'single_task_preds' in locals():
        print("\n--- Processing Single-Task Ensembles ---")
        for fold, targets_dict in single_task_preds.items():
            if len(targets_dict) != len(target_columns):
                print(f"WARNING: Missing single-task models for {fold}. Skipping.")
                continue

            # Combine predictions from kcat, KM, Ki models
            final_preds = torch.cat([
                targets_dict['kcat'], 
                targets_dict['KM'], 
                targets_dict['Ki']
            ], dim=1)
            
            # Get true targets for this fold (they are all the same)
            fold_num = fold.split('_')[-1]
            valid_path = VALID_DATA_PATH_TEMPLATE.format(fold_num=fold_num)
            dm = AstraDataModule(data_paths={'valid': valid_path}, **BASE_CONFIG_FOR_DATA)
            dm.setup('validate')
            dataloader = dm.val_dataloader()
            _, true_targets = get_predictions(AstraModule.load_from_checkpoint(list(ensemble_groups.values())[0][0]), dataloader, DEVICE) # a bit of a hack to get targets

            group_metrics = {'fold': fold, 'task_type': 'single_task', 'recomp': 'direct'}
            for i, param_name in enumerate(target_columns):
                preds_i = final_preds[:, i].to(DEVICE)
                targets_i = true_targets[:, i].to(DEVICE)
                
                mask = ~torch.isnan(targets_i)
                if mask.sum() > 0:
                    metrics = metric_collections[param_name](preds_i[mask], targets_i[mask])
                    metrics_renamed = {f"valid/{param_name}_{k}": v.item() for k, v in metrics.items()}
                    group_metrics.update(metrics_renamed)
            
            all_results.append(group_metrics)
    
    # --- 5. Save Final Results ---
    results_df = pd.DataFrame(all_results)
    # Add your new tag
    results_df['Tags'] = 'top-5-ensemble' 
    
    print("\n--- FINAL ENSEMBLE METRICS ---")
    print(results_df)
    
    output_path = RESULTS_CSV.parent / "ensemble_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")