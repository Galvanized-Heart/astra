import click
from pathlib import Path
from astra.analysis.loader import RunLoader
from astra.analysis.plotting import (
    calculate_metrics, 
    plot_performance_overview, 
    plot_architecture_comparison,
    plot_comprehensive_parity,
    plot_residuals,
    plot_clustered_bar_charts,
    print_summary_table,
)
from astra.constants import PROJECT_ROOT

@click.command()
@click.option('--tags', multiple=True, default=['5fcv'], help='W&B tags to filter runs')
@click.option('--output', default='results/figures/5cv_report', help='Output directory')
def main(tags, output):
    entity = "lmse-university-of-toronto"
    project = "astra"
    
    out_dir = PROJECT_ROOT / output
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Analysis ---")
    loader = RunLoader(entity, project)
    runs = loader.get_runs(list(tags))
    
    if not runs:
        print("No runs found.")
        return

    print("Loading prediction data...")
    df = loader.load_predictions(runs)
    if df.empty: return
        
    targets = ['kcat', 'KM', 'Ki']
    
    print("Calculating metrics...")
    metrics_df = calculate_metrics(df, targets)
    metrics_df.to_csv(out_dir / "full_metrics.csv", index=False)

    print("Generating Visualizations...")
    
    # 1. Performance Overview
    plot_performance_overview(metrics_df, out_dir)
    
    # 2. Architecture Comparison
    plot_architecture_comparison(metrics_df, out_dir)
    
    # 3. Comprehensive Parity Plots (The requested Grid with Colorbar)
    plot_comprehensive_parity(df, targets, out_dir)
    
    # 4. Residual Plots (New Evaluation angle)
    plot_residuals(df, targets, out_dir)

    # 5. Clustered Bar Charts (Architecture on X, Tasks in Legend)
    plot_clustered_bar_charts(metrics_df, out_dir)
    
    # 6. Summary Table
    print_summary_table(metrics_df)
    
    print("\nAnalysis Complete.")

if __name__ == '__main__':
    main()