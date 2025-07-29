"""
End-to-End Dataset Splitting via Sequence Clustering with MMseqs2.

This script provides a comprehensive workflow to:
1. Read a CSV file containing biological sequences.
2. Cluster the sequences using MMseqs2 to group similar items.
3. Split the data into train/validation/test sets based on these clusters,
   preventing data leakage and ensuring robust model evaluation.
4. Generate a "capped" histogram to clearly visualize the distribution of
   the most common cluster sizes.
5. Create a CSV report detailing all clusters that exceed the specified size cap.
6. Analyze the distribution and overlap of kinetic parameters (kcat, KM, Ki) 
   across the dataset and splits using statistical summaries and Venn diagrams.
7. Optionally create N-fold cross-validation splits with parameter analysis.
"""

import os
import random
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Optional import for Venn diagrams
try:
    from matplotlib_venn import venn3, venn3_circles
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
    print("Warning: matplotlib-venn not available. Venn diagrams will be skipped.")
    print("Install with: pip install matplotlib-venn")

# ==============================================================================
# SECTION 1: Core MMseqs2 & File Helpers
# ==============================================================================

def create_fasta_from_csv(csv_path: str, seq_col: str, fasta_path: str):
    """Creates a FASTA file from a specified column in a CSV file."""
    df = pd.read_csv(csv_path)
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{idx}\n{row[seq_col]}\n")


def cluster_and_split(fasta_path: str, seq_id: float, split_ratios: dict, 
                      coverage: float, cov_mode: int, threads: int, 
                      cluster_mode: int, seed: int):
    """Runs MMseqs2 to cluster sequences and assigns them to splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/DB"
        cluster_path = f"{tmpdir}/clu"
        tsv_path = f"{tmpdir}/clusters.tsv"
        
        cmds = [
            ["mmseqs", "createdb", fasta_path, db_path],
            ["mmseqs", "cluster", db_path, cluster_path, f"{tmpdir}/tmp",
             "--min-seq-id", str(seq_id), "-c", str(coverage), "--cov-mode", str(cov_mode), 
             "--threads", str(threads), "--cluster-mode", str(cluster_mode)],
            ["mmseqs", "createtsv", db_path, db_path, cluster_path, tsv_path]
        ]
        
        for cmd in cmds:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"MMseqs2 failed at command: {' '.join(cmd)}\nStderr: {e.stderr}")
                raise
        
        clusters = {}
        with open(tsv_path) as f:
            for line in f:
                rep, member = line.strip().split('\t')
                clusters.setdefault(rep, []).append(member)

        cluster_reps = list(clusters.keys())
        rng = random.Random(seed)
        rng.shuffle(cluster_reps)
        
        n = len(cluster_reps)
        train_end = int(split_ratios['train'] * n)
        valid_end = train_end + int(split_ratios['valid'] * n)
        
        splits = {
            'train': cluster_reps[:train_end],
            'valid': cluster_reps[train_end:valid_end],
            'test': cluster_reps[valid_end:]
        }

        split_map = {}
        cluster_id_map = {}
        for split_name, reps in splits.items():
            for rep in reps:
                for member in clusters[rep]:
                    cluster_id_map[member] = rep
                    split_map[member] = split_name

        num_clusters = len(clusters)
        total_members = sum(len(members) for members in clusters.values())
        average_size = total_members / num_clusters if num_clusters > 0 else 0
        stats = {
            'num_clusters': num_clusters,
            'average_cluster_size': f"{average_size:.2f}",
            'sequences_clustered': total_members,
        }
        return split_map, cluster_id_map, stats, clusters


# ==============================================================================
# SECTION 2: Kinetic Parameter Analysis & Visualization Helpers
# ==============================================================================

def analyze_kinetic_parameters(df: pd.DataFrame, kinetic_cols: list = None):
    """
    Analyzes the distribution of kinetic parameters in the dataset.
    
    Args:
        df (pd.DataFrame): The dataframe to analyze
        kinetic_cols (list): List of kinetic parameter column names. 
                           Defaults to ['kcat', 'km', 'ki']
    
    Returns:
        dict: Dictionary containing analysis results
    """
    if kinetic_cols is None:
        kinetic_cols = ['kcat', 'KM', 'Ki']
    
    total_rows = len(df)
    results = {
        'total_samples': total_rows,
        'parameter_counts': {},
        'parameter_percentages': {},
        'samples_with_any_param': 0,
        'samples_with_all_params': 0
    }
    
    # Check which columns actually exist in the dataframe
    existing_cols = [col for col in kinetic_cols if col in df.columns]
    
    if not existing_cols:
        print(f"Warning: None of the kinetic parameter columns {kinetic_cols} found in dataframe")
        return results
    
    # Count non-null values for each parameter
    for col in existing_cols:
        non_null_count = df[col].notna().sum()
        results['parameter_counts'][col] = non_null_count
        results['parameter_percentages'][col] = (non_null_count / total_rows) * 100 if total_rows > 0 else 0
    
    # Count samples with any parameter
    if existing_cols:
        has_any_param = df[existing_cols].notna().any(axis=1).sum()
        results['samples_with_any_param'] = has_any_param
        results['samples_with_any_param_percentage'] = (has_any_param / total_rows) * 100 if total_rows > 0 else 0
        
        # Count samples with all parameters
        has_all_params = df[existing_cols].notna().all(axis=1).sum()
        results['samples_with_all_params'] = has_all_params
        results['samples_with_all_params_percentage'] = (has_all_params / total_rows) * 100 if total_rows > 0 else 0
    
    return results


def print_kinetic_analysis(analysis_results: dict, dataset_name: str = "Dataset"):
    """
    Prints a formatted report of kinetic parameter analysis.
    
    Args:
        analysis_results (dict): Results from analyze_kinetic_parameters
        dataset_name (str): Name of the dataset for the report header
    """
    print(f"\n--- {dataset_name} Kinetic Parameter Analysis ---")
    print(f"Total samples: {analysis_results['total_samples']:,}")
    
    if analysis_results['parameter_counts']:
        print("\nParameter availability:")
        for param, count in analysis_results['parameter_counts'].items():
            percentage = analysis_results['parameter_percentages'][param]
            print(f"  - {param.upper()}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\nSamples with any parameter: {analysis_results['samples_with_any_param']:,} ({analysis_results['samples_with_any_param_percentage']:.1f}%)")
        print(f"Samples with all parameters: {analysis_results['samples_with_all_params']:,} ({analysis_results['samples_with_all_params_percentage']:.1f}%)")
    else:
        print("No kinetic parameters found in dataset")
    print("-" * 50)


def analyze_kinetic_distribution_by_split(df: pd.DataFrame, kinetic_cols: list = None):
    """
    Analyzes kinetic parameter distribution across train/valid/test splits.
    
    Args:
        df (pd.DataFrame): Dataframe with 'split' column and kinetic parameters
        kinetic_cols (list): List of kinetic parameter column names
    
    Returns:
        dict: Analysis results by split
    """
    if kinetic_cols is None:
        kinetic_cols = ['kcat', 'KM', 'Ki']
    
    splits = df['split'].unique()
    split_analysis = {}
    
    print("\n--- Kinetic Parameter Distribution by Split ---")
    
    for split in ['train', 'valid', 'test']:  # Ensure consistent order
        if split in splits:
            split_df = df[df['split'] == split]
            split_analysis[split] = analyze_kinetic_parameters(split_df, kinetic_cols)
            print_kinetic_analysis(split_analysis[split], f"{split.capitalize()} Split")
            
            # Also analyze overlap for each split
            overlap_results = analyze_kinetic_parameter_overlap(split_df, kinetic_cols)
            if overlap_results:
                print_overlap_analysis(overlap_results, f"{split.capitalize()} Split")
                split_analysis[split]['overlap'] = overlap_results
    
    return split_analysis


def analyze_kinetic_parameter_overlap(df: pd.DataFrame, kinetic_cols: list = None):
    """
    Analyzes the overlap of kinetic parameters (which samples have which combinations).
    
    Args:
        df (pd.DataFrame): The dataframe to analyze
        kinetic_cols (list): List of exactly 3 kinetic parameter column names for Venn diagram.
                           Defaults to ['kcat', 'KM', 'Ki']
    
    Returns:
        dict: Dictionary containing overlap analysis results
    """
    if kinetic_cols is None:
        kinetic_cols = ['kcat', 'KM', 'Ki']
    
    # Check which columns actually exist in the dataframe
    existing_cols = [col for col in kinetic_cols if col in df.columns]
    
    if len(existing_cols) < 2:
        print(f"Warning: Need at least 2 kinetic parameter columns for overlap analysis. Found: {existing_cols}")
        return {}
    
    # Limit to first 3 columns for Venn diagram compatibility
    if len(existing_cols) > 3:
        print(f"Note: Using first 3 kinetic parameters for Venn diagram: {existing_cols[:3]}")
        existing_cols = existing_cols[:3]
    
    total_rows = len(df)
    results = {
        'total_samples': total_rows,
        'parameters': existing_cols,
        'individual_counts': {},
        'overlap_counts': {},
        'venn_data': {}
    }
    
    # Count individual parameters (sets A, B, C)
    sets = {}
    for col in existing_cols:
        has_param = df[col].notna()
        sets[col] = set(df[has_param].index)
        results['individual_counts'][col] = len(sets[col])
    
    if len(existing_cols) == 2:
        # Two-way overlap analysis
        col1, col2 = existing_cols
        set1, set2 = sets[col1], sets[col2]
        
        overlap_12 = set1 & set2
        only_1 = set1 - set2
        only_2 = set2 - set1
        
        results['overlap_counts'] = {
            f'{col1}_only': len(only_1),
            f'{col2}_only': len(only_2),
            f'{col1}_{col2}': len(overlap_12),
            'neither': total_rows - len(set1 | set2)
        }
        
    elif len(existing_cols) == 3:
        # Three-way overlap analysis
        col1, col2, col3 = existing_cols
        set1, set2, set3 = sets[col1], sets[col2], sets[col3]
        
        # Calculate all possible overlaps for Venn diagram
        only_1 = set1 - set2 - set3
        only_2 = set2 - set1 - set3  
        only_3 = set3 - set1 - set2
        overlap_12_not3 = (set1 & set2) - set3
        overlap_13_not2 = (set1 & set3) - set2
        overlap_23_not1 = (set2 & set3) - set1
        overlap_123 = set1 & set2 & set3
        
        results['overlap_counts'] = {
            f'{col1}_only': len(only_1),
            f'{col2}_only': len(only_2), 
            f'{col3}_only': len(only_3),
            f'{col1}_{col2}_only': len(overlap_12_not3),
            f'{col1}_{col3}_only': len(overlap_13_not2),
            f'{col2}_{col3}_only': len(overlap_23_not1),
            f'{col1}_{col2}_{col3}': len(overlap_123),
            'none': total_rows - len(set1 | set2 | set3)
        }
        
        # Prepare data for matplotlib-venn (order: only_1, only_2, overlap_12_not3, only_3, overlap_13_not2, overlap_23_not1, overlap_123)
        results['venn_data'] = {
            'subsets': (
                len(only_1),           # A only
                len(only_2),           # B only  
                len(overlap_12_not3),  # A&B only
                len(only_3),           # C only
                len(overlap_13_not2),  # A&C only
                len(overlap_23_not1),  # B&C only
                len(overlap_123)       # A&B&C
            ),
            'set_labels': existing_cols
        }
    
    return results


def print_overlap_analysis(overlap_results: dict, dataset_name: str = "Dataset"):
    """
    Prints a formatted report of kinetic parameter overlap analysis.
    
    Args:
        overlap_results (dict): Results from analyze_kinetic_parameter_overlap
        dataset_name (str): Name of the dataset for the report header
    """
    if not overlap_results:
        return
        
    print(f"\n--- {dataset_name} Kinetic Parameter Overlap Analysis ---")
    print(f"Total samples: {overlap_results['total_samples']:,}")
    print(f"Parameters analyzed: {', '.join(overlap_results['parameters'])}")
    
    print(f"\nIndividual parameter counts:")
    for param, count in overlap_results['individual_counts'].items():
        percentage = (count / overlap_results['total_samples']) * 100 if overlap_results['total_samples'] > 0 else 0
        print(f"  - {param.upper()}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"\nOverlap analysis:")
    for overlap_type, count in overlap_results['overlap_counts'].items():
        percentage = (count / overlap_results['total_samples']) * 100 if overlap_results['total_samples'] > 0 else 0
        overlap_label = overlap_type.replace('_', ' & ').upper().replace(' ONLY', ' only').replace('NONE', 'No parameters')
        print(f"  - {overlap_label}: {count:,} samples ({percentage:.1f}%)")
    
    print("-" * 60)


def create_kinetic_parameter_venn_diagram(overlap_results: dict, output_path: str, dataset_name: str = "Dataset"):
    """
    Creates a Venn diagram showing the overlap of kinetic parameters.
    
    Args:
        overlap_results (dict): Results from analyze_kinetic_parameter_overlap
        output_path (str): Path to save the Venn diagram
        dataset_name (str): Name of the dataset for the plot title
    """
    if not HAS_VENN:
        print("Warning: matplotlib-venn not available. Skipping Venn diagram creation.")
        return
        
    if not overlap_results or 'venn_data' not in overlap_results:
        print("Warning: No Venn diagram data available. Skipping visualization.")
        return
    
    if len(overlap_results['parameters']) != 3:
        print(f"Warning: Venn diagram requires exactly 3 parameters. Found {len(overlap_results['parameters'])}. Skipping visualization.")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Define distinct colors for each set
    colors = {
        'set1': '#FF9999',  # Light red
        'set2': '#99FF99',  # Light green
        'set3': '#9999FF',  # Light blue
        'alpha': 0.4        # Transparency for better overlap visibility
    }
    
    # Create the Venn diagram with custom colors
    venn_data = overlap_results['venn_data']
    v = venn3(subsets=venn_data['subsets'], 
              set_labels=venn_data['set_labels'],
              set_colors=(colors['set1'], colors['set2'], colors['set3']),
              alpha=colors['alpha'])
    
    # Customize the appearance
    if v is not None:
        # Add thin black edge to each circle for better definition
        c = venn3_circles(subsets=venn_data['subsets'], linestyle='-', linewidth=1)
        
        # Customize text appearance
        for text in v.set_labels:
            if text is not None:
                text.set_fontweight('bold')
                text.set_fontsize(11)
        
        # Make the numbers more visible
        for text in v.subset_labels:
            if text is not None:
                text.set_fontsize(10)
                text.set_fontweight('bold')
    
    # Add title and labels
    param_names = [param.upper() for param in overlap_results['parameters']]
    plt.title(f'{dataset_name}\nKinetic Parameter Overlap: {", ".join(param_names)}', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add summary statistics as text
    total_samples = overlap_results['total_samples']
    summary_text = f"Total samples: {total_samples:,}\n"
    
    for param, count in overlap_results['individual_counts'].items():
        pct = (count / total_samples) * 100 if total_samples > 0 else 0
        summary_text += f"{param.upper()}: {count:,} ({pct:.1f}%)\n"
    
    # Add all three parameters count
    if 'venn_data' in overlap_results:
        all_three_count = venn_data['subsets'][6]  # A&B&C intersection
        all_three_pct = (all_three_count / total_samples) * 100 if total_samples > 0 else 0
        summary_text += f"\nAll three: {all_three_count:,} ({all_three_pct:.1f}%)"
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                        edgecolor='gray', linewidth=1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_nfold_cross_validation_splits(train_df: pd.DataFrame, n_folds: int, output_dir: str, 
                                        seed: int = 42, kinetic_cols: list = None):
    """
    Creates N-fold cross-validation splits with proper train/test pairs for each fold.
    For each fold, one part serves as test and the remaining parts as train.
    
    Args:
        train_df (pd.DataFrame): Training dataframe to split into folds
        n_folds (int): Number of folds for cross-validation
        output_dir (str): Directory to save the fold CSV files
        seed (int): Random seed for reproducible splitting
        kinetic_cols (list): List of kinetic parameter column names for analysis
    
    Returns:
        dict: Information about the created folds
    """
    print(f"\n--- Creating {n_folds}-Fold Cross-Validation Splits ---")
    
    if n_folds < 2:
        raise ValueError("Number of folds must be at least 2")
    
    if len(train_df) < n_folds:
        raise ValueError(f"Training set has {len(train_df)} samples, which is less than {n_folds} folds")
    
    # Shuffle the training data
    shuffled_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Use sklearn's KFold for proper cross-validation splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_info = {}
    
    # Create CV folds directory
    cv_output_dir = os.path.join(output_dir, 'cv_folds')
    os.makedirs(cv_output_dir, exist_ok=True)
    
    fold_analyses = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(shuffled_df)):
        # Create train and test sets for this fold
        fold_train_df = shuffled_df.iloc[train_idx].copy()
        fold_test_df = shuffled_df.iloc[test_idx].copy()
        
        # Save train and test sets
        train_filename = f"fold_{fold_idx}_train.csv"
        test_filename = f"fold_{fold_idx}_test.csv"
        
        train_path = os.path.join(cv_output_dir, train_filename)
        test_path = os.path.join(cv_output_dir, test_filename)
        
        fold_train_df.to_csv(train_path, index=False)
        fold_test_df.to_csv(test_path, index=False)
        
        # Store fold information
        fold_info[f'fold_{fold_idx}'] = {
            'train_path': train_path,
            'test_path': test_path,
            'train_count': len(fold_train_df),
            'test_count': len(fold_test_df)
        }
        
        print(f"  -> Fold {fold_idx}: Train {len(fold_train_df):,} samples, Test {len(fold_test_df):,} samples")
        
        # Analyze kinetic parameters for this fold if requested
        if kinetic_cols:
            train_analysis = analyze_kinetic_parameters(fold_train_df, kinetic_cols)
            test_analysis = analyze_kinetic_parameters(fold_test_df, kinetic_cols)
            
            fold_analyses[f'fold_{fold_idx}'] = {
                'train': train_analysis,
                'test': test_analysis
            }
            
            # Analyze overlap for this fold
            train_overlap = analyze_kinetic_parameter_overlap(fold_train_df, kinetic_cols)
            test_overlap = analyze_kinetic_parameter_overlap(fold_test_df, kinetic_cols)
            
            # Create Venn diagrams for this fold
            if train_overlap and 'venn_data' in train_overlap:
                train_venn_path = os.path.join(cv_output_dir, f"fold_{fold_idx}_train_overlap.png")
                create_kinetic_parameter_venn_diagram(train_overlap, train_venn_path, f"Fold {fold_idx} Train")
            
            if test_overlap and 'venn_data' in test_overlap:
                test_venn_path = os.path.join(cv_output_dir, f"fold_{fold_idx}_test_overlap.png")
                create_kinetic_parameter_venn_diagram(test_overlap, test_venn_path, f"Fold {fold_idx} Test")
    
    # Print detailed kinetic parameter analysis across folds
    if kinetic_cols and fold_analyses:
        print(f"\n--- Kinetic Parameter Distribution Across {n_folds} Folds ---")
        
        # Calculate original parameter percentages for comparison
        original_analysis = analyze_kinetic_parameters(train_df, kinetic_cols)
        
        param_deviations = {param: [] for param in kinetic_cols if param in train_df.columns}
        
        for fold_name, fold_data in fold_analyses.items():
            fold_num = fold_name.split('_')[1]
            print(f"\nFold {fold_num}:")
            print(f"  Train: {fold_data['train']['total_samples']:,} samples")
            print(f"  Test:  {fold_data['test']['total_samples']:,} samples")
            
            for param in kinetic_cols:
                if param in train_df.columns:
                    train_pct = fold_data['train']['parameter_percentages'].get(param, 0)
                    test_pct = fold_data['test']['parameter_percentages'].get(param, 0)
                    original_pct = original_analysis['parameter_percentages'].get(param, 0)
                    
                    train_dev = abs(train_pct - original_pct)
                    test_dev = abs(test_pct - original_pct)
                    
                    param_deviations[param].extend([train_dev, test_dev])
                    
                    print(f"    {param.upper()} - Train: {train_pct:.1f}% (dev: {train_dev:.1f}%), Test: {test_pct:.1f}% (dev: {test_dev:.1f}%)")
        
        # Print overall deviation summary
        print(f"\n--- Overall CV Parameter Distribution Quality ---")
        for param, deviations in param_deviations.items():
            if deviations:  # Only if we have data for this parameter
                avg_deviation = np.mean(deviations)
                std_deviation = np.std(deviations)
                print(f"  {param.upper()}: Avg deviation {avg_deviation:.1f}% ± {std_deviation:.1f}%")
    
    print(f"\n✓ Created {n_folds} cross-validation folds with total {len(train_df):,} samples")
    print(f"✓ Fold files saved in: {cv_output_dir}")
    
    return fold_info


# ==============================================================================
# SECTION 3: Visualization Helpers
# ==============================================================================

def plot_cluster_size_distribution_capped(clusters: dict, output_path: str, xlim_max: int):
    """Generates a histogram of cluster sizes with a capped x-axis."""
    if not clusters:
        return

    cluster_sizes = pd.Series([len(members) for members in clusters.values()])
    plt.figure(figsize=(12, 7))
    bins = np.arange(1, xlim_max + 2)
    plt.hist(cluster_sizes, bins=bins, alpha=0.8, label='Cluster Frequency')
    
    mean_size = cluster_sizes.mean()
    plt.axvline(mean_size, color='r', linestyle='--', linewidth=2, label=f'Overall Mean Size: {mean_size:.2f}')
    
    plt.xlim(0, xlim_max)
    plt.title(f'Distribution of Cluster Sizes (View Capped at Size {xlim_max})')
    plt.xlabel(f'Number of Sequences per Cluster (Capped at {xlim_max})')
    plt.ylabel('Frequency (Number of Clusters)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_and_plot_clusters(clusters: dict, size_threshold: int, output_dir: str):
    """
    Analyzes cluster sizes, creating a capped plot and a report for large clusters.
    """
    large_clusters_info = [
        {'representative_id': rep, 'cluster_size': len(members)}
        for rep, members in clusters.items() if len(members) > size_threshold
    ]

    report_path = os.path.join(output_dir, 'large_clusters_report.csv')
    if large_clusters_info:
        large_clusters_df = pd.DataFrame(large_clusters_info).sort_values('cluster_size', ascending=False)
        large_clusters_df.to_csv(report_path, index=False)
        print(f"-> Found {len(large_clusters_df)} clusters larger than {size_threshold}. Report saved to: {report_path}")
    else:
        print(f"-> No clusters found larger than {size_threshold}.")

    plot_path = os.path.join(output_dir, f'cluster_dist_capped_at_{size_threshold}.png')
    plot_cluster_size_distribution_capped(clusters, plot_path, size_threshold)
    print(f"-> Capped distribution plot saved to: {plot_path}")


# ==============================================================================
# SECTION 4: Main Orchestration Function
# ==============================================================================

def mmseqs2_split_data_into_files(
    input_csv_path: str,
    output_dir: str,
    seq_col: str = "protein_sequence",
    seq_id: float = 0.8,
    split_ratios: dict = None,
    coverage: float = 0.8,
    cov_mode: int = 0,
    cluster_mode: int = 0,
    threads: int = 4,
    seed: int = 42,
    cluster_size_cap: int = 50,
    kinetic_cols: list = None,
    n_folds: int = None
):
    """
    Orchestrates the full data splitting and analysis workflow.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_dir (str): Directory for all output files.
        seq_col (str): Name of the column containing sequences.
        seq_id (float): Sequence identity threshold for clustering.
        split_ratios (dict): Ratios for train/valid/test splits.
        coverage (float): Coverage threshold for clustering.
        cov_mode (int): Coverage mode for MMseqs2.
        cluster_mode (int): Cluster mode for MMseqs2.
        threads (int): Number of CPU threads for MMseqs2.
        seed (int): Random seed for reproducible splitting.
        cluster_size_cap (int): Threshold for reporting large clusters and
                                for capping the visualization plot.
        kinetic_cols (list): List of kinetic parameter column names to analyze.
                           Defaults to ['kcat', 'KM', 'Ki'].
        n_folds (int): Number of folds for cross-validation of training data.
                      If None, no cross-validation splits are created.
    """
    print("--- Starting Data Splitting and Analysis Process ---")
    
    # --- 1. Validation and Setup ---
    if not os.path.exists(input_csv_path): raise FileNotFoundError(f"Input not found: {input_csv_path}")
    if not shutil.which("mmseqs"): raise OSError("MMseqs2 not found in PATH.")
    if split_ratios is None: split_ratios = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    if abs(sum(split_ratios.values()) - 1.0) > 1e-6: raise ValueError("Split ratios must sum to 1.0.")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv_path)

    # --- 2. Analyze Original Dataset Kinetic Parameters ---
    print("2. Analyzing original dataset kinetic parameter distribution...")
    original_analysis = analyze_kinetic_parameters(df, kinetic_cols)
    print_kinetic_analysis(original_analysis, "Original Dataset")

    # --- 3. Create FASTA and Run Clustering ---
    with tempfile.NamedTemporaryFile(mode='w', suffix=".fasta", delete=False) as fasta_file:
        fasta_path = fasta_file.name
    try:
        print("3. Creating temporary FASTA file...")
        create_fasta_from_csv(input_csv_path, seq_col, fasta_path)

        print("4. Clustering sequences with MMseqs2...")
        split_map, cluster_map, stats, clusters = cluster_and_split(
            fasta_path, seq_id, split_ratios, coverage, cov_mode, threads, cluster_mode, seed
        )
    finally:
        if os.path.exists(fasta_path): os.remove(fasta_path)

    # --- 5. Analyze and Visualize Cluster Distribution ---
    print(f"5. Analyzing cluster distribution with a size cap of {cluster_size_cap}...")
    analyze_and_plot_clusters(clusters, cluster_size_cap, output_dir)
        
    # --- 6. Assign Splits and Save Data Files ---
    print("6. Assigning splits and saving partitioned CSV files...")
    df['cluster_id'] = df.index.astype(str).map(cluster_map)
    df['split'] = df.index.astype(str).map(split_map)
    
    if df['split'].isnull().sum() > 0:
        print(f"-> Warning: {df['split'].isnull().sum()} sequences were unclustered. Assigning to 'train'.")
        df['split'].fillna('train', inplace=True)

    output_files = {}
    for split_name in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split_name].copy().drop(columns=['split'])
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(output_path, index=False)
        output_files[split_name] = {'path': output_path, 'count': len(split_df)}

    # --- 7. Analyze Kinetic Parameter Distribution by Split ---
    print("7. Analyzing kinetic parameter distribution across splits...")
    split_analysis = analyze_kinetic_distribution_by_split(df, kinetic_cols)
    
    # Create Venn diagrams for each split
    for split_name in ['train', 'valid', 'test']:
        if split_name in split_analysis and 'overlap' in split_analysis[split_name]:
            overlap_data = split_analysis[split_name]['overlap']
            if overlap_data and 'venn_data' in overlap_data:
                venn_path = os.path.join(output_dir, f"{split_name}_kinetic_overlap.png")
                create_kinetic_parameter_venn_diagram(overlap_data, venn_path, f"{split_name.capitalize()} Split")

    # --- 8. Analyze Kinetic Parameter Overlap ---
    print("8. Analyzing kinetic parameter overlap...")
    overlap_results = analyze_kinetic_parameter_overlap(df, kinetic_cols)
    print_overlap_analysis(overlap_results, "Original Dataset")
    create_kinetic_parameter_venn_diagram(overlap_results, os.path.join(output_dir, "kinetic_parameter_overlap.png"), "Original Dataset")

    # --- 9. Create N-Fold Cross-Validation Splits (Optional) ---
    fold_info = None
    if n_folds is not None and n_folds > 1:
        print(f"9. Creating {n_folds}-fold cross-validation splits from training data...")
        train_df = df[df['split'] == 'train'].copy().drop(columns=['split'])
        fold_info = create_nfold_cross_validation_splits(
            train_df, n_folds, output_dir, seed, kinetic_cols
        )

    # --- 10. Final Report ---
    print("\n--- Process Complete ---")
    print("Clustering Statistics:")
    for key, value in stats.items(): print(f"  - {key.replace('_', ' ').title()}: {value}")
    
    print("\nOutput Files:")
    for split_name, info in output_files.items(): print(f"  - {info['path']} ({info['count']} rows)")
    print(f"  - {os.path.join(output_dir, f'cluster_dist_capped_at_{cluster_size_cap}.png')}")
    # if os.path.exists(os.path.join(output_dir, 'large_clusters_report.csv')):
    #     print(f"  - {os.path.join(output_dir, 'large_clusters_report.csv')}")
    # if os.path.exists(os.path.join(output_dir, "kinetic_parameter_overlap.png")):
    #     print(f"  - {os.path.join(output_dir, "kinetic_parameter_overlap.png")}")
    
    # Report split-specific overlap diagrams
    for split_name in ['train', 'valid', 'test']:
        split_venn_path = os.path.join(output_dir, f"{split_name}_kinetic_overlap.png")
        if os.path.exists(split_venn_path):
            print(f"  - {split_venn_path}")
    
    # Report cross-validation folds if created
    if fold_info:
        print(f"\nCross-Validation Folds ({n_folds}-fold):")
        total_fold_samples = 0
        for fold_name, info in fold_info.items():
            print(f"  - {info['train_path']} ({info['train_count']} rows)")
            print(f"  - {info['test_path']} ({info['test_count']} rows)")
            total_fold_samples += info['train_count'] + info['test_count']
            
            # Report fold-specific overlap diagrams if they exist
            fold_idx = fold_name.split('_')[1]
            train_venn = os.path.join(output_dir, 'cv_folds', f"fold_{fold_idx}_train_overlap.png")
            test_venn = os.path.join(output_dir, 'cv_folds', f"fold_{fold_idx}_test_overlap.png")
            if os.path.exists(train_venn):
                print(f"  - {train_venn}")
            if os.path.exists(test_venn):
                print(f"  - {test_venn}")
                
        print(f"  Total samples in folds: {total_fold_samples:,}")
    
    print("------------------------\n")

# ==============================================================================
# SECTION 5: Example Usage
# ==============================================================================

if __name__ == '__main__':
    # --- Create a dummy CSV for demonstration ---
    print("Creating a dummy 'input.csv' for demonstration...")
    np.random.seed(42)
    small = [1] * 100 + [2] * 20 + list(np.random.randint(3, 50, 20))
    medium = [55, 67, 120]
    all_sizes = small + medium
    sequences = []
    for i, size in enumerate(all_sizes):
        base_seq = ''.join(random.choices('ACGT', k=30))
        for j in range(size):
            # Create slight variations for each member
            mutated_seq = list(base_seq)
            mutated_seq[j % 30] = random.choice('ACGT')
            sequences.append(''.join(mutated_seq))
            
    dummy_df = pd.DataFrame({'SEQ': sequences})
    
    # Add dummy kinetic parameters to test overlap functionality
    n_samples = len(sequences)
    random.seed(42)
    np.random.seed(42)
    
    # Create realistic distributions of kinetic parameters
    # Some samples have kcat, some have KM, some have Ki, with varying overlaps
    kcat_values = []
    km_values = []
    ki_values = []
    
    for i in range(n_samples):
        # Create realistic missing data patterns
        has_kcat = random.random() < 0.6  # 60% have kcat
        has_km = random.random() < 0.4    # 40% have KM
        has_ki = random.random() < 0.3    # 30% have Ki
        
        kcat_values.append(np.random.lognormal(2, 1) if has_kcat else np.nan)
        km_values.append(np.random.lognormal(0, 1) if has_km else np.nan)
        ki_values.append(np.random.lognormal(1, 1) if has_ki else np.nan)
    
    dummy_df['kcat'] = kcat_values
    dummy_df['KM'] = km_values
    dummy_df['Ki'] = ki_values
    
    dummy_input_csv = "input_for_splitting.csv"
    dummy_df.to_csv(dummy_input_csv, index=False)
    
    OUTPUT_DIRECTORY = "final_split_output"
    
    # --- Run the main function if MMseqs2 is available ---
    if shutil.which("mmseqs"):
        mmseqs2_split_data_into_files(
            input_csv_path=dummy_input_csv,
            output_dir=OUTPUT_DIRECTORY,
            seq_col="SEQ",
            seq_id=0.7, # Lower identity to find more clusters
            cluster_size_cap=50, # Set the cap for reporting/plotting
            kinetic_cols=['kcat', 'KM', 'Ki'],  # Enable kinetic parameter analysis
            n_folds=5  # Enable 5-fold cross-validation to test overlap analysis
        )
    else:
        print("\n" + "="*50)
        print("WARNING: MMseqs2 executable not found. Main script was skipped.")
        print("Please install MMseqs2 and ensure it's in your system's PATH.")
        print("="*50 + "\n")