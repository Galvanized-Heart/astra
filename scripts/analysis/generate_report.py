import click
from pathlib import Path
from astra.analysis.loader import RunLoader
from astra.analysis.plotting import plot_parity, plot_metrics_comparison
from astra.constants import PROJECT_ROOT

@click.command()
@click.option('--tags', multiple=True, default=['5fcv'], help='W&B tags to filter runs (e.g., 5fcv)')
@click.option('--output', default='results/figures', help='Output directory for plots')
def main(tags, output):
    """
    Generates analysis plots for Astra experiments.
    """
    entity = "lmse-university-of-toronto"
    project = "astra"
    
    out_dir = PROJECT_ROOT / output
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Analysis for tags: {tags} ---")
    
    # Load Data
    loader = RunLoader(entity, project)
    runs = loader.get_runs(list(tags))
    
    if not runs:
        print("No runs found. Exiting.")
        return

    df = loader.load_predictions(runs)
    
    if df.empty:
        print("No prediction data loaded (check if files exist locally).")
        return
        
    print(f"Loaded {len(df)} total predictions across {df['run_id'].nunique()} runs.")
    
    # Define Targets
    targets = ['kcat', 'KM', 'Ki']
    
    # Generate Plots
    print("Generating Parity Plots...")
    plot_parity(df, targets, out_dir)
    
    print("Generating Metric Comparisons...")
    plot_metrics_comparison(df, targets, out_dir)
    
    print("Analysis Complete.")

if __name__ == '__main__':
    main()