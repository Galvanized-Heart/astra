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
    """
    try:
        hist_cols = list(run.summary.keys())
    except Exception:
        return None 

    col_map = {}
    train_cols = [c for c in hist_cols if 'train' in c and 'loss' in c and 'epoch' in c]
    if train_cols: col_map['train_loss'] = train_cols[0]
    
    valid_cols = [c for c in hist_cols if 'valid' in c and 'loss' in c and 'epoch' in c]
    if valid_cols: col_map['valid_loss'] = valid_cols[0]
        
    return col_map if 'train_loss' in col_map and 'valid_loss' in col_map else None

def load_loss_history(loader: 'RunLoader', runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    """
    Robustly fetches and reshapes W&B history.
    """
    all_epoch_data = []
    
    for run in tqdm(runs, desc="Fetching & Processing History"):
        col_map = _get_loss_history_keys(run)
        if col_map is None:
            # Silent skip or low verbosity to avoid clutter
            continue
            
        train_key = col_map['train_loss']
        valid_key = col_map['valid_loss']

        try:
            # Fetch loss individually
            train_hist = run.history(pandas=True, keys=[train_key], samples=500).drop(columns=['_step'], errors='ignore')
            valid_hist = run.history(pandas=True, keys=[valid_key], samples=500).drop(columns=['_step'], errors='ignore')
            
            train_hist = train_hist.rename(columns={train_key: 'train_loss_epoch'})
            valid_hist = valid_hist.rename(columns={valid_key: 'valid_loss_epoch'})

            epoch_summary = train_hist.join(valid_hist)
            epoch_summary = epoch_summary.reset_index().rename(columns={'index': 'epoch'})
            epoch_summary = epoch_summary.dropna(subset=['train_loss_epoch', 'valid_loss_epoch'])

        except Exception as e:
            continue

        if epoch_summary.empty:
            continue

        arch, mode = loader._derive_metadata(run.config)
        epoch_summary['architecture'] = arch
        epoch_summary['experiment_mode'] = mode
        
        all_epoch_data.append(epoch_summary)
        
    if not all_epoch_data:
        return pd.DataFrame()
        
    return pd.concat(all_epoch_data, ignore_index=True)

def plot_loss_curves(df: pd.DataFrame, save_dir: Path):
    """
    Generates two grids of learning curves:
    1. Log Scale
    2. Linear Scale (Zoomed 0.0 - 3.0)
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
    
    # Define settings for the two desired plots
    plot_settings = [
        {
            "name": "log",
            "scale": "log",
            "ylim": None, # Log scale handles ranges well automatically
            "title": "Training vs. Validation Loss (Log Scale)"
        },
        {
            "name": "linear",
            "scale": "linear", 
            "ylim": (0.0, 3.0), # Zoom in to cut off initial spikes
            "title": "Training vs. Validation Loss (Linear Scale)"
        }
    ]

    for settings in plot_settings:
        g = sns.relplot(
            data=melted_df,
            x='epoch',
            y='Loss',
            hue='Metric Type',
            col='experiment_mode',
            row='architecture',
            kind='line',
            errorbar='sd',
            facet_kws={'margin_titles': True, 'sharey': True}, 
            height=3.5,
            aspect=1.2,
            palette={'Train': '#1f77b4', 'Validation': '#ff7f0e'},
            row_order=final_arch_order,
            col_order=final_mode_order
        )
        
        g.set_axis_labels("Epoch", "Loss (MSE)")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        
        # Apply Scale
        g.set(yscale=settings["scale"])

        # Apply Y-Limits if defined (Crucial for the Linear plot)
        if settings["ylim"]:
            g.set(ylim=settings["ylim"])

        g.fig.suptitle(settings["title"], y=1.03)

        save_path = save_dir / f"loss_learning_curves_{settings['name']}.png"
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {settings['name']} scale curves to {save_path}")
        
        plt.close() # Close figure to free memory before next loop

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
    if not runs: 
        return
                
    history_df = load_loss_history(loader, runs)

    if history_df.empty:
        print("\nERROR: No valid epoch-level loss history found for any runs.")
        return
        
    plot_loss_curves(history_df, out_dir)
        
    print("\nOverfitting analysis complete.")

if __name__ == '__main__':
    main()