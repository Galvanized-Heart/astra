import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_parity(df: pd.DataFrame, target_cols: list, save_dir: Path):
    """
    Generates parity plots (Predicted vs Actual).
    Expects df to have columns: 'kcat', 'kcat_pred', 'architecture', etc.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Melt dataframe for faceting
    plot_data = []
    for col in target_cols:
        pred_col = f"{col}_pred"
        if col in df.columns and pred_col in df.columns:
            temp = df[[col, pred_col, 'architecture', 'experiment_mode']].copy()
            temp = temp.rename(columns={col: 'Actual', pred_col: 'Predicted'})
            temp['Target'] = col
            plot_data.append(temp)
    
    if not plot_data:
        return

    full_df = pd.concat(plot_data)

    # Create a FacetGrid
    g = sns.FacetGrid(full_df, col="Target", row="architecture", hue="experiment_mode", 
                      sharex=False, sharey=False, height=4, aspect=1)
    g.map(sns.scatterplot, "Actual", "Predicted", alpha=0.3, s=15)
    
    # Add diagonal lines
    for ax in g.axes.flat:
        # Get limits
        low = min(ax.get_xlim()[0], ax.get_ylim()[0])
        high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([low, high], [low, high], 'k--', lw=1)

    g.add_legend()
    g.savefig(save_dir / "parity_plots.png", dpi=300)
    print(f"Saved parity plots to {save_dir}")

def plot_metrics_comparison(df: pd.DataFrame, target_cols: list, save_dir: Path):
    """
    Calculates Pearson R per fold/arch/mode and plots bar chart.
    """
    from scipy.stats import pearsonr
    
    results = []
    
    # Group by fold to calculate metrics per fold (for error bars)
    groups = df.groupby(['architecture', 'experiment_mode', 'fold'])
    
    for (arch, mode, fold), group in groups:
        for target in target_cols:
            pred_col = f"{target}_pred"
            if target not in group.columns or pred_col not in group.columns:
                continue
                
            # Drop NaNs
            clean = group[[target, pred_col]].dropna()
            if len(clean) < 2: 
                continue
                
            r, _ = pearsonr(clean[target], clean[pred_col])
            rmse = np.sqrt(((clean[target] - clean[pred_col]) ** 2).mean())
            
            results.append({
                'Architecture': arch,
                'Mode': mode,
                'Fold': fold,
                'Target': target,
                'Pearson R': r,
                'RMSE': rmse
            })
            
    res_df = pd.DataFrame(results)
    
    # Plot Pearson
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x="Target", y="Pearson R", hue="Architecture", 
                errorbar='sd', capsize=.1, palette="viridis")
    plt.title("Model Performance by Architecture (Pearson R)")
    plt.ylim(0, 1.0)
    plt.savefig(save_dir / "metrics_pearson.png", dpi=300)
    plt.close()

    # Plot Comparison of Experiment Modes (e.g. Recomp vs Direct)
    # We filter for a specific architecture to see mode impact, or use catplot
    g = sns.catplot(data=res_df, x="Target", y="Pearson R", hue="Mode", col="Architecture",
                    kind="bar", errorbar='sd', capsize=.1, height=5, aspect=0.8)
    g.savefig(save_dir / "mode_comparison.png", dpi=300)
    print(f"Saved metric plots to {save_dir}")