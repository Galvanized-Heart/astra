import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

# --- CONSTANTS ---
ARCH_ORDER = ["Linear", "Conv1D", "Self-Attn", "Cross-Attn"]
MODE_ORDER = ["Single Task", "Multi-Task (Direct)", "Multi-Task (Basic)", "Multi-Task (Advanced)"]

def get_present_order(available_items, desired_order):
    """Helper to maintain consistent ordering."""
    available_set = set(available_items)
    final_order = [x for x in desired_order if x in available_set]
    final_order.extend([x for x in available_items if x not in final_order])
    return final_order

def calculate_metrics(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """Calculates metrics per fold."""
    results = []
    groups = df.groupby(['architecture', 'experiment_mode', 'fold'])
    
    print(f"Calculating metrics for {len(groups)} experimental groups...")
    
    for (arch, mode, fold), group in groups:
        for target in target_cols:
            pred_col = f"{target}_pred"
            if pred_col not in group.columns or group[pred_col].isna().all(): continue
            
            clean = group[[target, pred_col]].dropna()
            if len(clean) < 10: continue
            
            y_true = clean[target].values
            y_pred = clean[pred_col].values
            
            p_r, _ = pearsonr(y_true, y_pred)
            s_r, _ = spearmanr(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            results.append({
                'Architecture': arch, 'Mode': mode, 'Fold': fold, 'Target': target,
                'Pearson': p_r, 'Spearman': s_r, 'R2': r2, 'RMSE': rmse, 'MAE': mae
            })
            
    return pd.DataFrame(results)

def plot_comprehensive_parity(df: pd.DataFrame, target_cols: list, save_dir: Path):
    """
    Generates one plot per Architecture.
    Grid: Rows = Modes, Columns = Targets.
    Includes Colorbar and Metrics Annotation.
    """
    save_dir = save_dir / "parity_plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter Architectures based on order
    archs = get_present_order(df['architecture'].unique(), ARCH_ORDER)

    for arch in archs:
        print(f"Generating parity grid for {arch}...")
        
        # Filter for this architecture
        arch_df = df[df['architecture'] == arch].copy()
        
        # Prepare data for FacetGrid (Melting targets)
        plot_data = []
        stats_map = {}

        for target in target_cols:
            pred_col = f"{target}_pred"
            if pred_col not in arch_df.columns: continue
            
            # Select relevant columns
            temp = arch_df[['experiment_mode', target, pred_col]].dropna()
            if temp.empty: continue
            
            # Calculate stats per Mode/Target for annotation
            # Note: aggregating all folds here for the visualization
            for mode, grp in temp.groupby('experiment_mode'):
                r2 = r2_score(grp[target], grp[pred_col])
                pr, _ = pearsonr(grp[target], grp[pred_col])
                sr, _ = spearmanr(grp[target], grp[pred_col])
                stats_map[(mode, target)] = (r2, pr, sr)

            temp = temp.rename(columns={target: 'Actual', pred_col: 'Predicted', 'experiment_mode': 'Mode'})
            temp['Target'] = target
            plot_data.append(temp)

        if not plot_data: continue
        full_data = pd.concat(plot_data)

        # Enforce Mode Order
        modes_present = get_present_order(full_data['Mode'].unique(), MODE_ORDER)
        
        # --- Plotting ---
        g = sns.FacetGrid(
            full_data, 
            row="Mode", 
            col="Target", 
            row_order=modes_present,
            col_order=target_cols,
            height=3.5, 
            aspect=1, 
            sharex=False, 
            sharey=False,
            margin_titles=True
        )

        # We need a mappable for the colorbar, so we grab the output of hexbin
        mappable = None

        def hexbin_plot(x, y, **kwargs):
            nonlocal mappable
            # vmin=1 ensures 0-count bins are empty (white)
            hb = plt.hexbin(x, y, gridsize=40, cmap="magma_r", mincnt=1, norm=mcolors.LogNorm())
            if mappable is None: mappable = hb

        g.map(hexbin_plot, "Actual", "Predicted")

        # Formatting
        for i, ax in enumerate(g.axes.flat):
            # 1. Identity Line
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # If axis is auto-scaled to (0,1) because empty, skip
            if xlim == (0.0, 1.0) and ylim == (0.0, 1.0): continue

            low = min(xlim[0], ylim[0])
            high = max(xlim[1], ylim[1])
            ax.plot([low, high], [low, high], 'k--', lw=1.5, alpha=0.6)
            ax.set_aspect('equal')

            # 2. Annotations
            # FacetGrid iterates row-major. 
            # We need to map linear index 'i' back to (Mode, Target)
            num_cols = len(g.col_names)
            row_idx = i // num_cols
            col_idx = i % num_cols
            
            curr_mode = g.row_names[row_idx]
            curr_target = g.col_names[col_idx]

            if (curr_mode, curr_target) in stats_map:
                r2, pr, sr = stats_map[(curr_mode, curr_target)]
                txt = (f"$R^2={r2:.2f}$\n$r={pr:.2f}$\n$\\rho={sr:.2f}$")
                ax.text(0.05, 0.95, txt, transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"))

        g.set_axis_labels("Experimental ($log_{10}$)", "Predicted ($log_{10}$)")
        g.fig.suptitle(f"Parity Plots: {arch}", y=1.02, fontsize=16)

        # --- Add Colorbar ---
        # Adjust layout to make room on the right
        g.fig.subplots_adjust(right=0.85)
        # Create a new axes for the colorbar
        cbar_ax = g.fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        cbar = g.fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Density (Log Scale)')

        save_path = save_dir / f"parity_{arch}.png"
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()

def plot_residuals(df: pd.DataFrame, target_cols: list, save_dir: Path):
    """
    Plots the distribution of residuals (Predicted - Actual).
    """
    save_dir = save_dir / "residual_plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate Residuals
    plot_data = []
    for target in target_cols:
        pred_col = f"{target}_pred"
        if pred_col in df.columns:
            temp = df.copy()
            temp['Residual'] = temp[pred_col] - temp[target]
            temp['Target'] = target
            plot_data.append(temp[['architecture', 'experiment_mode', 'Target', 'Residual']].dropna())
            
    if not plot_data: return
    full_df = pd.concat(plot_data)
    
    # Enforce order
    full_df['architecture'] = pd.Categorical(full_df['architecture'], categories=get_present_order(full_df['architecture'], ARCH_ORDER), ordered=True)
    full_df['experiment_mode'] = pd.Categorical(full_df['experiment_mode'], categories=get_present_order(full_df['experiment_mode'], MODE_ORDER), ordered=True)

    # Plot
    try:
        g = sns.FacetGrid(full_df, row="architecture", col="Target", hue="experiment_mode", 
                          height=3, aspect=1.5, sharex=False, palette="viridis")
        
        g.map(sns.kdeplot, "Residual", fill=True, alpha=0.1, linewidth=2)
        
        # --- FIX: Iterate axes to draw reference line instead of using .map() ---
        for ax in g.axes.flat:
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
        
        g.add_legend(title="Mode")
        g.set_axis_labels("Residual ($y_{pred} - y_{true}$)", "Density")
        g.fig.suptitle("Error Distribution (Residuals)", y=1.02)
        
        g.savefig(save_dir / "residual_distributions.png", dpi=300, bbox_inches='tight')
        print(f"Saved residual plots to {save_dir}")
    except Exception as e:
        print(f"Error plotting residuals: {e}")

def plot_performance_overview(metrics_df: pd.DataFrame, save_dir: Path):
    """Generates bar charts for every metric calculated."""
    if metrics_df.empty: return

    metrics_to_plot = ['Pearson', 'Spearman', 'R2', 'RMSE', 'MAE']
    
    # Apply standard ordering
    final_arch_order = get_present_order(metrics_df['Architecture'].unique(), ARCH_ORDER)
    final_mode_order = get_present_order(metrics_df['Mode'].unique(), MODE_ORDER)

    plot_dir = save_dir / "metrics_bar_charts"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics_to_plot:
        try:
            g = sns.catplot(
                data=metrics_df, 
                kind="bar",
                x="Mode", 
                y=metric, 
                hue="Architecture",
                col="Target",
                hue_order=final_arch_order,
                order=final_mode_order,
                palette="viridis",
                errorbar="sd", 
                capsize=.1,
                height=5, 
                aspect=1.0,
                sharey=(metric != 'RMSE' and metric != 'MAE') # Share Y for correlations
            )
            
            g.despine(left=True)
            g.set_axis_labels("", metric)
            g.set_titles("{col_name}")
            g.set_xticklabels(rotation=30, ha='right')
            
            if metric in ['R2', 'Pearson', 'Spearman']:
                for ax in g.axes.flat:
                    ax.axhline(0, color="k", linewidth=1, alpha=0.5)

            g.savefig(plot_dir / f"{metric}_overview.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
    
    print(f"Saved metric bar charts to {plot_dir}")

def plot_architecture_comparison(metrics_df: pd.DataFrame, save_dir: Path):
    """Generates aggregated boxplots for every metric."""
    if metrics_df.empty: return

    metrics_to_plot = ['Pearson', 'Spearman', 'R2', 'RMSE', 'MAE']
    final_arch_order = get_present_order(metrics_df['Architecture'].unique(), ARCH_ORDER)
    
    plot_dir = save_dir / "architecture_comparisons"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        try:
            sns.boxplot(
                data=metrics_df,
                x="Target",
                y=metric,
                hue="Architecture",
                hue_order=final_arch_order,
                palette="Set2"
            )
            plt.title(f"{metric} Stability (Aggregated across Modes)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            plt.savefig(plot_dir / f"{metric}_boxplot.png", dpi=300, bbox_inches='tight')
        except Exception:
            pass
        finally:
            plt.close()
            
    print(f"Saved architecture boxplots to {plot_dir}")

def print_summary_table(metrics_df: pd.DataFrame):
    """
    Prints a publication-ready summary table aggregating mean +/- std across folds.
    """
    if metrics_df.empty: return

    summary = metrics_df.groupby(['Architecture', 'Mode', 'Target']).agg(
        {
            'Pearson': ['mean', 'std'],
            'Spearman': ['mean', 'std'],
            'R2': ['mean', 'std'],
            'RMSE': ['mean', 'std']
        }
    ).reset_index()

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    print("\n" + "="*100)
    print("FINAL MODEL COMPARISON (Aggregated across 5 Folds)")
    print("="*100)
    
    # Sort combos by ARCH_ORDER and MODE_ORDER
    combos = summary[['Architecture', 'Mode']].drop_duplicates().copy()
    
    # Apply sorting
    combos['Architecture'] = pd.Categorical(combos['Architecture'], categories=get_present_order(combos['Architecture'], ARCH_ORDER), ordered=True)
    combos['Mode'] = pd.Categorical(combos['Mode'], categories=get_present_order(combos['Mode'], MODE_ORDER), ordered=True)
    combos = combos.sort_values(['Architecture', 'Mode'])
    
    for _, row in combos.iterrows():
        arch = row['Architecture']
        mode = row['Mode']
        
        subset = summary[(summary['Architecture'] == arch) & (summary['Mode'] == mode)]
        
        print(f"\nModel: {arch} | Mode: {mode}")
        print("-" * 90)
        print(f"{'Metric':<15} | {'kcat':<20} | {'KM':<20} | {'Ki':<20}")
        print("-" * 90)
        
        def get_fmt(tgt, metric):
            row = subset[subset['Target'] == tgt]
            if row.empty: return "N/A"
            mean = row[f"{metric}_mean"].values[0]
            std = row[f"{metric}_std"].values[0]
            return f"{mean:.4f} ± {std:.4f}"

        for metric in ['Pearson', 'Spearman', 'R2', 'RMSE']:
            kcat_val = get_fmt('kcat', metric)
            km_val = get_fmt('KM', metric)
            ki_val = get_fmt('Ki', metric)
            print(f"{metric:<15} | {kcat_val:<20} | {km_val:<20} | {ki_val:<20}")
    print("="*100 + "\n")



def plot_clustered_bar_charts(metrics_df: pd.DataFrame, out_dir: Path):
    """
    Creates clustered bar charts with Architecture on the X-axis.
    Generates two sets of charts for each Metric (R2, RMSE, etc.):
      1. Clustered by Target (kcat, KM, Ki in the legend)
      2. Clustered by Experiment Mode (Single vs Multi-task in the legend)
    """
    # Create sub-directory to keep things organized
    plot_dir = out_dir / "clustered_bar_charts"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Use capitalized column names as output by calculate_metrics
    target_col = 'Target' 
    if target_col not in metrics_df.columns:
        print("Warning: Could not find Target column. Skipping clustered bar charts.")
        return

    # Identify metadata columns vs metric columns dynamically
    meta_cols = ['Architecture', 'Mode', 'Fold', target_col]
    existing_meta = [c for c in meta_cols if c in metrics_df.columns]
    metric_cols = [c for c in metrics_df.columns if c not in existing_meta]

    # Melt DataFrame for Seaborn
    melted_df = metrics_df.melt(
        id_vars=existing_meta,
        value_vars=metric_cols,
        var_name='Metric',
        value_name='Value'
    )
    
    # Define orders to keep charts perfectly consistent
    arch_order = ["Linear", "Conv1D", "Self-Attn", "Cross-Attn"]
    mode_order = ["Single Task", "Multi-Task (Direct)", "Multi-Task (Basic)", "Multi-Task (Advanced)"]
    
    final_arch_order = [a for a in arch_order if a in melted_df['Architecture'].unique()]
    final_mode_order = [m for m in mode_order if m in melted_df['Mode'].unique()]
    
    # Plot separately for each Metric (R2, RMSE, etc.)
    for metric in melted_df['Metric'].unique():
        df_metric = melted_df[melted_df['Metric'] == metric]

        # =========================================================
        # VIEW 1: X = Architecture, Legend = Target (kcat, KM, Ki)
        # =========================================================
        g1 = sns.catplot(
            data=df_metric,
            x='Architecture',
            y='Value',
            hue=target_col,
            col='Mode',            # Separate panels for Single vs Multi-task
            col_order=final_mode_order,
            kind='bar',
            errorbar='sd',
            capsize=0.1,
            height=4.5,
            aspect=1.2,
            palette='Set2',
            order=final_arch_order
        )
        
        g1.fig.suptitle(f"{metric} Comparison: Architecture vs. Target Task", y=1.05)
        g1.set_axis_labels("Architecture", metric)
        g1.set_titles("{col_name}")
        
        # Save View 1
        save_path_1 = plot_dir / f"clustered_arch_by_target_{metric.lower()}.png"
        g1.savefig(save_path_1, dpi=300, bbox_inches='tight')
        plt.close(g1.fig)

        # =========================================================
        # VIEW 2: X = Architecture, Legend = Experiment Mode
        # =========================================================
        g2 = sns.catplot(
            data=df_metric,
            x='Architecture',
            y='Value',
            hue='Mode',
            hue_order=final_mode_order,
            col=target_col,        # Separate panels for kcat, KM, Ki
            kind='bar',
            errorbar='sd',
            capsize=0.1,
            height=4.5,
            aspect=1.2,
            palette='viridis',
            order=final_arch_order
        )
        
        g2.fig.suptitle(f"{metric} Comparison: Architecture vs. Experiment Mode", y=1.05)
        g2.set_axis_labels("Architecture", metric)
        g2.set_titles("Target: {col_name}")

        # Save View 2
        save_path_2 = plot_dir / f"clustered_arch_by_mode_{metric.lower()}.png"
        g2.savefig(save_path_2, dpi=300, bbox_inches='tight')
        plt.close(g2.fig)
        
    print(f"Saved clustered architecture comparison charts to {plot_dir}")