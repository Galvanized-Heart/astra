import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import List
import wandb

from astra.analysis.loader import RunLoader
from astra.constants import PROJECT_ROOT

def _get_loss_history_keys(run: wandb.apis.public.Run) -> dict | None:
    """
    Tries to get column names by fetching a minimal history sample.
    This is necessary because not all runs might use the exact same key names.
    """
    try:
        # Fetch a small sample to see all available keys in the history
        hist_cols = run.history(pandas=True, samples=1).columns.tolist()
    except Exception:
        return None 

    col_map = {}
    # Find train loss key
    train_cols = [c for c in hist_cols if 'train' in c and 'loss' in c and 'epoch' in c]
    if train_cols: col_map['train_loss'] = train_cols[0]
    
    # Find validation loss key
    valid_cols = [c for c in hist_cols if 'valid' in c and 'loss' in c and 'epoch' in c]
    if valid_cols: col_map['valid_loss'] = valid_cols[0]
        
    return col_map if 'train_loss' in col_map and 'valid_loss' in col_map else None

def load_loss_history(loader: 'RunLoader', runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    """
    Robustly fetches and reshapes W&B history based on individual key fetching
    and index-based epoch alignment.
    """
    all_epoch_data = []
    
    for run in tqdm(runs, desc="Fetching & Processing History"):
        
        # MINIMAL CHANGE: Find keys dynamically instead of hardcoding
        col_map = _get_loss_history_keys(run)
        if col_map is None:
            tqdm.write(f"Skipping run {run.name}: Could not find required train/valid loss keys.")
            continue
            
        train_key = col_map['train_loss']
        valid_key = col_map['valid_loss']

        try:
            # Fetch loss individually, dropping the now-unnecessary _step column
            train_hist = run.history(pandas=True, keys=[train_key], samples=500).drop(columns=['_step'], errors='ignore')
            valid_hist = run.history(pandas=True, keys=[valid_key], samples=500).drop(columns=['_step'], errors='ignore')
            
            # Rename columns to a standard name before joining
            train_hist = train_hist.rename(columns={train_key: 'train_loss_epoch'})
            valid_hist = valid_hist.rename(columns={valid_key: 'valid_loss_epoch'})

            # Combine dataframes based on their index (which corresponds to epoch for complete runs)
            epoch_summary = train_hist.join(valid_hist)

            # Use the DataFrame index to create the 'epoch' column
            epoch_summary = epoch_summary.reset_index().rename(columns={'index': 'epoch'})
            
            # Final check to remove any rows that might be incomplete after the join
            epoch_summary = epoch_summary.dropna(subset=['train_loss_epoch', 'valid_loss_epoch'])

        except Exception as e:
            tqdm.write(f"Skipping run {run.name} due to an error: {e}")
            continue

        if epoch_summary.empty:
            continue

        # Add metadata
        arch, mode = loader._derive_metadata(run.config)
        epoch_summary['architecture'] = arch
        epoch_summary['experiment_mode'] = mode
        
        all_epoch_data.append(epoch_summary)
        
    if not all_epoch_data:
        return pd.DataFrame()
        
    return pd.concat(all_epoch_data, ignore_index=True)

def plot_loss_curves(df: pd.DataFrame, save_dir: Path):
    """
    Generates a grid of learning curves (Train vs. Valid Loss) with shared Y-axes.
    """
    melted_df = df.melt(
        id_vars=['epoch', 'architecture', 'experiment_mode'], 
        value_vars=['train_loss_epoch', 'valid_loss_epoch'],
        var_name='Metric Type', 
        value_name='Loss'
    )
    melted_df['Metric Type'] = melted_df['Metric Type'].apply(
        lambda x: 'Train' if 'train' in x else 'Validation'
    )
    
    mode_order = ["Single Task", "Multi-Task (Direct)", "Multi-Task (Basic)", "Multi-Task (Advanced)"]
    arch_order = ["Linear", "Conv1D", "Self-Attn", "Cross-Attn"]
    
    final_mode_order = [m for m in mode_order if m in melted_df['experiment_mode'].unique()]
    final_arch_order = [a for a in arch_order if a in melted_df['architecture'].unique()]
    
    g = sns.relplot(
        data=melted_df,
        x='epoch',
        y='Loss',
        hue='Metric Type',
        col='experiment_mode',
        row='architecture',
        kind='line',
        errorbar='sd',
        # --- CHANGE: Set sharey to True ---
        facet_kws={'margin_titles': True, 'sharey': True}, 
        height=3.5,
        aspect=1.2,
        palette={'Train': '#1f77b4', 'Validation': '#ff7f0e'},
        row_order=final_arch_order,
        col_order=final_mode_order
    )
    
    g.set_axis_labels("Epoch", "Loss (MSE)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    
    # Since you are using a log scale, sharing the axis is very helpful 
    # for seeing order-of-magnitude differences between models.
    g.set(yscale="log")
    
    # OPTIONAL: If early-epoch spikes make the rest of the graph look flat, 
    # you can manually set limits, for example:
    # g.set(ylim=(df['train_loss_epoch'].min() * 0.9, df['train_loss_epoch'].max()))

    g.fig.suptitle("Training vs. Validation Loss (Shared Scale)", y=1.03)

    save_path = save_dir / "loss_learning_curves.png"
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss learning curves to {save_path}")
    plt.close()

@click.command()
@click.option('--tags', multiple=True, default=['5fcv'], help='W&B tags to filter runs.')
@click.option('--output', default='results/figures/overfitting_analysis', help='Output directory.')
def main(tags, output):
    """
    Analyzes overfitting by fetching training history from W&B
    and plotting train vs. validation loss curves.
    """
    entity = "lmse-university-of-toronto"
    project = "astra"

    out_dir = PROJECT_ROOT / output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- Starting Overfitting Analysis (Loss Curves) ---")

    loader = RunLoader(entity, project)
    runs = loader.get_runs(list(tags))
    if not runs: return
                
    history_df = load_loss_history(loader, runs)

    if history_df.empty:
        print("\nERROR: No valid epoch-level loss history found for any runs.")
        return
        
    plot_loss_curves(history_df, out_dir)
        
    print("\nOverfitting analysis complete.")

if __name__ == '__main__':
    main()