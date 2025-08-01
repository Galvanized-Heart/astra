"""
End-to-End Dataset Splitting via Sequence Clustering with MMseqs2.

This script provides a comprehensive workflow to:
1. Read a CSV file containing biological sequences.
2. Cluster the sequences using MMseqs2 to group similar items.
3. Generate a "capped" histogram to clearly visualize the distribution of
   the most common cluster sizes.
4. Create a CSV report detailing all clusters that exceed the specified size cap.
5. Analyze the distribution and overlap of kinetic parameters (kcat, KM, Ki) 
   across the dataset using statistical summaries and Venn diagrams.
6. Optionally create N-fold cross-validation splits with parameter analysis.
"""

import os
import random
import shutil
import subprocess
import tempfile
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from matplotlib_venn import venn3, venn3_circles


# ==============================================================================
# SECTION 1: Core MMseqs2 & File Helpers
# ==============================================================================

def create_fasta_from_csv(csv_path: str, seq_col: str, fasta_path: str):
    """Creates a FASTA file from a specified column in a CSV file."""
    df = pd.read_csv(csv_path)
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{idx}\n{row[seq_col]}\n")


def cluster_sequences(fasta_path: str, seq_id: float, coverage: float, cov_mode: int, 
                     threads: int, cluster_mode: int):
    """Runs MMseqs2 to cluster sequences."""
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

        cluster_id_map = {}
        for rep, members in clusters.items():
            for member in members:
                cluster_id_map[member] = rep

        num_clusters = len(clusters)
        total_members = sum(len(members) for members in clusters.values())
        average_size = total_members / num_clusters if num_clusters > 0 else 0
        stats = {
            'num_clusters': num_clusters,
            'average_cluster_size': f"{average_size:.2f}",
            'sequences_clustered': total_members,
        }
        return cluster_id_map, stats, clusters


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
        
        # Prepare data for matplotlib-venn 
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
    
    print(f"\nOverlap analysis:")
    for overlap_type, count in overlap_results['overlap_counts'].items():
        percentage = (count / overlap_results['total_samples']) * 100 if overlap_results['total_samples'] > 0 else 0
        if overlap_type == 'none':
            overlap_label = 'No parameters'
        elif overlap_type.endswith('_only'):
            params = overlap_type[:-5] 
            overlap_label = params.replace('_', ' & ').upper() + ' only'
        else:
            overlap_label = overlap_type.replace('_', ' & ').upper()
        
        print(f"  - {overlap_label}: {count:,} samples ({percentage:.1f}%)")
    
    print("-" * 60)



def create_nfold_cross_validation_splits(df: pd.DataFrame, n_folds: int, output_dir: str, 
                                       seed: int = 42, kinetic_cols: list = None):
    """
    Creates N-fold cross-validation splits with proper train/valid pairs for each fold.
    For each fold, one part serves as valid and the remaining parts as train.
    
    Args:
        df (pd.DataFrame): Full dataframe to split into folds, including cluster information
        n_folds (int): Number of folds for cross-validation
        output_dir (str): Directory to save the fold CSV files
        seed (int): Random seed for reproducible splitting
        kinetic_cols (list): List of kinetic parameter column names for analysis
    """
    print(f"\n--- Creating {n_folds}-Fold Cross-Validation Splits ---")
    
    if n_folds < 2:
        raise ValueError("Number of folds must be at least 2")
    
    if len(df) < n_folds:
        raise ValueError(f"Dataset has {len(df)} samples, which is less than {n_folds} folds")
    
    # Get unique cluster IDs and shuffle them
    cluster_ids = df['cluster_id'].unique()
    rng = random.Random(seed)
    cluster_ids = list(cluster_ids)
    rng.shuffle(cluster_ids)
    
    # Split clusters into folds
    fold_size = len(cluster_ids) // n_folds
    fold_clusters = [cluster_ids[i:i + fold_size] for i in range(0, len(cluster_ids), fold_size)]
    
    # Adjust last fold to include any remaining clusters
    if len(fold_clusters) > n_folds:
        fold_clusters[-2].extend(fold_clusters[-1])
        fold_clusters.pop()
    
    # Create CV folds directory
    cv_output_dir = os.path.join(output_dir, 'cv_folds')
    os.makedirs(cv_output_dir, exist_ok=True)
    
    fold_info = {}
    fold_analyses = {}
    
    # Create folds based on clusters
    for fold_idx in range(n_folds):
        # Valid set is current fold's clusters
        valid_clusters = fold_clusters[fold_idx]
        # Train set is all other clusters
        train_clusters = [c for i, fold in enumerate(fold_clusters) if i != fold_idx for c in fold]
        
        # Create train and valid sets
        fold_valid_df = df[df['cluster_id'].isin(valid_clusters)].copy()
        fold_train_df = df[df['cluster_id'].isin(train_clusters)].copy()
        
        # Save train and valid sets
        train_filename = f"fold_{fold_idx}_train.csv"
        valid_filename = f"fold_{fold_idx}_valid.csv"
        
        train_path = os.path.join(cv_output_dir, train_filename)
        valid_path = os.path.join(cv_output_dir, valid_filename)
        
        fold_train_df.to_csv(train_path, index=False)
        fold_valid_df.to_csv(valid_path, index=False)
        
        # Store fold information
        fold_info[f'fold_{fold_idx}'] = {
            'train_path': train_path,
            'valid_path': valid_path,
            'train_count': len(fold_train_df),
            'valid_count': len(fold_valid_df),
            'train_clusters': len(train_clusters),
            'valid_clusters': len(valid_clusters)
        }
        
        print(f"  -> Fold {fold_idx}: Train {len(fold_train_df):,} samples ({len(train_clusters)} clusters), Valid {len(fold_valid_df):,} samples ({len(valid_clusters)} clusters)")
        
        # Analyze kinetic parameters for this fold if requested
        if kinetic_cols:
            train_analysis = analyze_kinetic_parameters(fold_train_df, kinetic_cols)
            valid_analysis = analyze_kinetic_parameters(fold_valid_df, kinetic_cols)
            
            fold_analyses[f'fold_{fold_idx}'] = {
                'train': train_analysis,
                'valid': valid_analysis
            }
            
            # Analyze overlap for this fold
            train_overlap = analyze_kinetic_parameter_overlap(fold_train_df, kinetic_cols)
            valid_overlap = analyze_kinetic_parameter_overlap(fold_valid_df, kinetic_cols)
            
            # Create Venn diagrams for this fold
            if train_overlap and 'venn_data' in train_overlap:
                create_kinetic_parameter_venn_diagram(train_overlap, cv_output_dir, f"Fold {fold_idx} Train")
            
            if valid_overlap and 'venn_data' in valid_overlap:
                create_kinetic_parameter_venn_diagram(valid_overlap, cv_output_dir, f"Fold {fold_idx} Valid")
    
    # Print detailed kinetic parameter analysis across folds
    if kinetic_cols and fold_analyses:
        print(f"\n--- Kinetic Parameter Distribution Across {n_folds} Folds ---")
        
        # Calculate original parameter percentages for comparison
        original_analysis = analyze_kinetic_parameters(df, kinetic_cols)
        
        param_deviations = {param: [] for param in kinetic_cols if param in df.columns}
        
        for fold_name, fold_data in fold_analyses.items():
            fold_num = fold_name.split('_')[1]
            print(f"\nFold {fold_num}:")
            print(f"  Train: {fold_data['train']['total_samples']:,} samples")
            print(f"  Valid:  {fold_data['valid']['total_samples']:,} samples")
            
            for param in kinetic_cols:
                if param in df.columns:
                    train_pct = fold_data['train']['parameter_percentages'].get(param, 0)
                    valid_pct = fold_data['valid']['parameter_percentages'].get(param, 0)
                    original_pct = original_analysis['parameter_percentages'].get(param, 0)
                    
                    train_dev = abs(train_pct - original_pct)
                    valid_dev = abs(valid_pct - original_pct)
                    
                    param_deviations[param].extend([train_dev, valid_dev])
                    
                    print(f"    {param.upper()} - Train: {train_pct:.1f}% (dev: {train_dev:.1f}%), Valid: {valid_pct:.1f}% (dev: {valid_dev:.1f}%)")
        
        # Print overall deviation summary
        print(f"\n--- Overall CV Parameter Distribution Quality ---")
        for param, deviations in param_deviations.items():
            if deviations:  # Only if we have data for this parameter
                avg_deviation = np.mean(deviations)
                std_deviation = np.std(deviations)
                print(f"  {param.upper()}: Avg deviation {avg_deviation:.1f}% ± {std_deviation:.1f}%")
    
    print(f"\nCreated {n_folds} cross-validation folds with total {len(df):,} samples")
    
    return fold_info


def calculate_kinetic_distribution(df: pd.DataFrame, kinetic_cols: list[str]) -> dict[str, float]:
    """Calculate the percentage of non-null values for each kinetic parameter."""
    distribution = {}
    if len(df) == 0:
        return {col: 0.0 for col in kinetic_cols if col in df.columns}
    
    for col in kinetic_cols:
        if col in df.columns:
            distribution[col] = (df[col].notna().sum() / len(df)) * 100
        else:
            distribution[col] = 0.0
    return distribution


def calculate_distribution_deviation(target_dist: dict[str, float], 
                                   current_dist: dict[str, float]) -> float:
    """Calculate the total absolute deviation from target distribution."""
    total_deviation = 0.0
    for param in target_dist:
        if param in current_dist:
            total_deviation += abs(target_dist[param] - current_dist[param])
    return total_deviation


def evaluate_fold_assignment(df: pd.DataFrame, 
                           fold_assignments: dict[str, int], 
                           n_folds: int, 
                           target_distribution: dict[str, float],
                           kinetic_cols: list[str]) -> tuple[float, dict]:
    """
    Evaluate how well the current fold assignment preserves kinetic parameter distribution.
    Returns total deviation and detailed fold statistics.
    """
    fold_stats = {}
    total_deviation = 0.0
    
    for fold_idx in range(n_folds):
        # Get samples assigned to this fold
        fold_clusters = [cluster for cluster, fold in fold_assignments.items() if fold == fold_idx]
        fold_df = df[df['cluster_id'].isin(fold_clusters)]
        
        # Calculate distribution for this fold
        fold_distribution = calculate_kinetic_distribution(fold_df, kinetic_cols)
        
        # Calculate deviation from target
        fold_deviation = calculate_distribution_deviation(target_distribution, fold_distribution)
        total_deviation += fold_deviation
        
        fold_stats[fold_idx] = {
            'size': len(fold_df),
            'distribution': fold_distribution,
            'deviation': fold_deviation
        }
    
    return total_deviation, fold_stats


def greedy_fold_assignment(df: pd.DataFrame, 
                          cluster_ids: list[str], 
                          n_folds: int, 
                          target_distribution: dict[str, float],
                          kinetic_cols: list[str],
                          max_iterations: int = 1000,
                          seed: int = 42) -> dict[str, int]:
    """
    Greedily assign clusters to folds to minimize kinetic parameter distribution deviation.
    """
    rng = random.Random(seed)
    
    # Initialize with random assignment
    fold_assignments = {}
    for i, cluster_id in enumerate(cluster_ids):
        fold_assignments[cluster_id] = i % n_folds
    
    # Calculate initial score
    best_score, _ = evaluate_fold_assignment(df, fold_assignments, n_folds, target_distribution, kinetic_cols)
    best_assignments = deepcopy(fold_assignments)
    
    print(f"  Initial distribution deviation: {best_score:.2f}")
    
    improvements = 0
    for iteration in range(max_iterations):
        # Try swapping clusters between random folds
        cluster1, cluster2 = rng.sample(cluster_ids, 2)
        
        if fold_assignments[cluster1] == fold_assignments[cluster2]:
            continue  # Skip if already in same fold
        
        # Try the swap
        original_fold1 = fold_assignments[cluster1]
        original_fold2 = fold_assignments[cluster2]
        
        fold_assignments[cluster1] = original_fold2
        fold_assignments[cluster2] = original_fold1
        
        # Evaluate new assignment
        new_score, _ = evaluate_fold_assignment(df, fold_assignments, n_folds, target_distribution, kinetic_cols)
        
        if new_score < best_score:
            best_score = new_score
            best_assignments = deepcopy(fold_assignments)
            improvements += 1
        else:
            # Revert the swap
            fold_assignments[cluster1] = original_fold1
            fold_assignments[cluster2] = original_fold2
        
        # Print progress every 200 iterations
        if (iteration + 1) % 200 == 0:
            print(f"  Iteration {iteration + 1}: Best deviation = {best_score:.2f} (improvements: {improvements})")
    
    print(f"  Final distribution deviation: {best_score:.2f} (total improvements: {improvements})")
    return best_assignments


def create_balanced_nfold_cross_validation_splits(df: pd.DataFrame, 
                                                n_folds: int, 
                                                output_dir: str,
                                                seed: int = 42, 
                                                kinetic_cols: list = None,
                                                max_iterations: int = 1000):
    """
    Creates N-fold cross-validation splits with kinetic parameter distribution constraints.
    Uses greedy search to minimize deviation from original dataset distribution.
    """
    print(f"\n--- Creating Balanced {n_folds}-Fold Cross-Validation Splits ---")
    
    if n_folds < 2:
        raise ValueError("Number of folds must be at least 2")
    
    if len(df) < n_folds:
        raise ValueError(f"Dataset has {len(df)} samples, which is less than {n_folds} folds")
    
    if kinetic_cols is None:
        kinetic_cols = ['kcat', 'KM', 'Ki']
    
    # Filter to existing columns
    existing_kinetic_cols = [col for col in kinetic_cols if col in df.columns]
    if not existing_kinetic_cols:
        print("Warning: No kinetic parameter columns found. Using standard random assignment.")
        return create_nfold_cross_validation_splits(df, n_folds, output_dir, seed, kinetic_cols)
    
    print(f"Optimizing distribution for parameters: {existing_kinetic_cols}")
    
    # Calculate target distribution from original dataset
    target_distribution = calculate_kinetic_distribution(df, existing_kinetic_cols)
    print("Target distribution:")
    for param, pct in target_distribution.items():
        print(f"  {param.upper()}: {pct:.1f}%")
    
    # Get unique cluster IDs
    cluster_ids = df['cluster_id'].unique().tolist()
    print(f"Total clusters to assign: {len(cluster_ids)}")
    
    # Use greedy search to find optimal fold assignments
    print(f"Running greedy optimization with {max_iterations} iterations...")
    fold_assignments = greedy_fold_assignment(
        df, cluster_ids, n_folds, target_distribution, existing_kinetic_cols, max_iterations, seed
    )
    
    # Create CV folds directory
    cv_output_dir = os.path.join(output_dir, 'cv_folds_balanced')
    os.makedirs(cv_output_dir, exist_ok=True)
    
    fold_info = {}
    fold_analyses = {}
    
    # Create folds based on optimized cluster assignments
    for fold_idx in range(n_folds):
        # Valid set is current fold's clusters
        valid_clusters = [cluster for cluster, fold in fold_assignments.items() if fold == fold_idx]
        # Train set is all other clusters
        train_clusters = [cluster for cluster, fold in fold_assignments.items() if fold != fold_idx]
        
        # Create train and valid sets
        fold_valid_df = df[df['cluster_id'].isin(valid_clusters)].copy()
        fold_train_df = df[df['cluster_id'].isin(train_clusters)].copy()
        
        # Save train and valid sets
        train_filename = f"fold_{fold_idx}_train.csv"
        valid_filename = f"fold_{fold_idx}_valid.csv"
        
        train_path = os.path.join(cv_output_dir, train_filename)
        valid_path = os.path.join(cv_output_dir, valid_filename)
        
        fold_train_df.to_csv(train_path, index=False)
        fold_valid_df.to_csv(valid_path, index=False)
        
        # Store fold information
        fold_info[f'fold_{fold_idx}'] = {
            'train_path': train_path,
            'valid_path': valid_path,
            'train_count': len(fold_train_df),
            'valid_count': len(fold_valid_df),
            'train_clusters': len(train_clusters),
            'valid_clusters': len(valid_clusters)
        }
        
        print(f"  -> Fold {fold_idx}: Train {len(fold_train_df):,} samples ({len(train_clusters)} clusters), Valid {len(fold_valid_df):,} samples ({len(valid_clusters)} clusters)")
        
        # Analyze kinetic parameters for this fold
        train_analysis = analyze_kinetic_parameters(fold_train_df, existing_kinetic_cols)
        valid_analysis = analyze_kinetic_parameters(fold_valid_df, existing_kinetic_cols)
        
        fold_analyses[f'fold_{fold_idx}'] = {
            'train': train_analysis,
            'valid': valid_analysis
        }
        
        # Analyze overlap for this fold
        train_overlap = analyze_kinetic_parameter_overlap(fold_train_df, existing_kinetic_cols)
        valid_overlap = analyze_kinetic_parameter_overlap(fold_valid_df, existing_kinetic_cols)
        
        # Create Venn diagrams for this fold
        if train_overlap and 'venn_data' in train_overlap:
            create_kinetic_parameter_venn_diagram(train_overlap, cv_output_dir, f"Fold {fold_idx} Train")
        
        if valid_overlap and 'venn_data' in valid_overlap:
            create_kinetic_parameter_venn_diagram(valid_overlap, cv_output_dir, f"Fold {fold_idx} Valid")
    
    # Print detailed kinetic parameter analysis across folds
    if fold_analyses:
        print(f"\n--- Balanced Kinetic Parameter Distribution Across {n_folds} Folds ---")
        
        # Calculate original parameter percentages for comparison
        original_analysis = analyze_kinetic_parameters(df, existing_kinetic_cols)
        
        param_deviations = {param: [] for param in existing_kinetic_cols}
        
        for fold_name, fold_data in fold_analyses.items():
            fold_num = fold_name.split('_')[1]
            print(f"\nFold {fold_num}:")
            print(f"  Train: {fold_data['train']['total_samples']:,} samples")
            print(f"  Valid:  {fold_data['valid']['total_samples']:,} samples")
            
            for param in existing_kinetic_cols:
                train_pct = fold_data['train']['parameter_percentages'].get(param, 0)
                valid_pct = fold_data['valid']['parameter_percentages'].get(param, 0)
                original_pct = original_analysis['parameter_percentages'].get(param, 0)
                
                train_dev = abs(train_pct - original_pct)
                valid_dev = abs(valid_pct - original_pct)
                
                param_deviations[param].extend([train_dev, valid_dev])
                
                print(f"    {param.upper()} - Train: {train_pct:.1f}% (dev: {train_dev:.1f}%), Valid: {valid_pct:.1f}% (dev: {valid_dev:.1f}%)")
        
        # Print overall deviation summary
        print(f"\n--- Overall Balanced CV Parameter Distribution Quality ---")
        for param, deviations in param_deviations.items():
            if deviations:  # Only if we have data for this parameter
                avg_deviation = np.mean(deviations)
                std_deviation = np.std(deviations)
                print(f"  {param.upper()}: Avg deviation {avg_deviation:.1f}% ± {std_deviation:.1f}%")
    
    print(f"\nCreated {n_folds} balanced cross-validation folds with total {len(df):,} samples")
    
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


    
def create_kinetic_parameter_venn_diagram(overlap_results: dict, output_path: str, dataset_name: str = "Dataset"):
    """
    Creates a Venn diagram showing the overlap of kinetic parameters.
    
    Args:
        overlap_results (dict): Results from analyze_kinetic_parameter_overlap
        output_path (str): Path to save the Venn diagram
        dataset_name (str): Name of the dataset for the plot title
    """
        
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

    # Save
    if dataset_name == "Original Dataset":
        # Save directly in the output directory
        filename = "original_dataset_venn.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight', facecolor='white')
    else:
        # Create venn_diagrams directory for fold-specific diagrams
        venn_dir = os.path.join(output_path, "venn_diagrams")
        os.makedirs(venn_dir, exist_ok=True)
        # Save figure
        filename = f"{dataset_name.replace(' ', '_').lower()}_venn.png"
        plt.savefig(os.path.join(venn_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ==============================================================================
# SECTION 4: Main Orchestration Function
# ==============================================================================

def mmseqs2_split_data_w_ratio_into_files(
    input_csv_path: str,
    output_dir: str,
    seq_col: str = "protein_sequence",
    seq_id: float = 0.8,
    coverage: float = 0.8,
    cov_mode: int = 0,
    cluster_mode: int = 0,
    threads: int = 4,
    seed: int = 42,
    cluster_size_cap: int = 50,
    kinetic_cols: list = None,
    n_folds: int = None,
    use_balanced_cv: bool = True,
    max_cv_iterations: int = 1000,
    compare_cv_methods: bool = True
):
    """
    Orchestrates the full data splitting and analysis workflow.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_dir (str): Directory for all output files.
        seq_col (str): Name of the column containing sequences.
        seq_id (float): Sequence identity threshold for clustering.
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
        use_balanced_cv (bool): Whether to use balanced cross-validation that 
                               preserves kinetic parameter distribution.
        max_cv_iterations (int): Maximum iterations for greedy search optimization.
        compare_cv_methods (bool): Whether to create both original and balanced CV
                                  for comparison.
    """
    print("--- Starting Data Clustering and Analysis Process ---")
    
    # --- 1. Validation and Setup ---
    if not os.path.exists(input_csv_path): raise FileNotFoundError(f"Input not found: {input_csv_path}")
    if not shutil.which("mmseqs"): raise OSError("MMseqs2 not found in PATH.")

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
        cluster_map, stats, clusters = cluster_sequences(
            fasta_path, seq_id, coverage, cov_mode, threads, cluster_mode
        )
    finally:
        if os.path.exists(fasta_path): os.remove(fasta_path)

    # --- 5. Analyze and Visualize Cluster Distribution ---
    print(f"5. Analyzing cluster distribution with a size cap of {cluster_size_cap}...")
    analyze_and_plot_clusters(clusters, cluster_size_cap, output_dir)
        
    # --- 6. Assign Cluster IDs ---
    print("6. Assigning cluster IDs...")
    df['cluster_id'] = df.index.astype(str).map(cluster_map)
    
    if df['cluster_id'].isnull().sum() > 0:
        print(f"-> Warning: {df['cluster_id'].isnull().sum()} sequences were unclustered.")

    # --- 7. Analyze Kinetic Parameter Overlap ---
    print("7. Analyzing kinetic parameter overlap...")
    overlap_results = analyze_kinetic_parameter_overlap(df, kinetic_cols)
    print_overlap_analysis(overlap_results, "Original Dataset")
    create_kinetic_parameter_venn_diagram(overlap_results, output_dir, "Original Dataset")
    
    # --- 8. N-Fold Cross-Validation Section ---
    fold_info = None
    if n_folds is not None and n_folds > 1:
        if compare_cv_methods:
            print(f"\n{'='*60}")
            print("CREATING BOTH ORIGINAL AND BALANCED CROSS-VALIDATION")
            print(f"{'='*60}")
            
            # First create original CV splits
            print(f"8a. Creating ORIGINAL {n_folds}-fold cross-validation splits...")
            
            # Get unique cluster IDs
            cluster_ids = df['cluster_id'].unique().tolist()
            print(f"Total clusters to assign: {len(cluster_ids)}")
            
            fold_info = create_nfold_cross_validation_splits(
                df, n_folds, output_dir, seed, kinetic_cols
            )
            
            if use_balanced_cv:
                # Then create balanced CV splits
                print(f"\n8b. Creating BALANCED {n_folds}-fold cross-validation splits...")
                balanced_fold_info = create_balanced_nfold_cross_validation_splits(
                    df, n_folds, output_dir, seed, kinetic_cols, max_cv_iterations
                )
                
                print(f"\n{'='*60}")
                print("CROSS-VALIDATION COMPARISON COMPLETE")
                print(f"{'='*60}")
                print("Original CV files saved in: cv_folds/")
                print("Balanced CV files saved in: cv_folds_balanced/")
                print("Compare the 'Overall CV Parameter Distribution Quality' metrics above!")
                print(f"{'='*60}")
        else:
            # Create only the requested type
            if use_balanced_cv:
                print(f"8. Creating BALANCED {n_folds}-fold cross-validation splits...")
                fold_info = create_balanced_nfold_cross_validation_splits(
                    df, n_folds, output_dir, seed, kinetic_cols, max_cv_iterations
                )
            else:
                print(f"8. Creating ORIGINAL {n_folds}-fold cross-validation splits...")
                fold_info = create_nfold_cross_validation_splits(
                    df, n_folds, output_dir, seed, kinetic_cols
                )

    # --- 9. Final Report ---
    print("\n--- Process Complete ---")
    print("Clustering Statistics:")
    for key, value in stats.items(): 
        print(f"  - {key.replace('_', ' ').title()}: {value}")
    
    print("\nOutput Files:")
    print(f"  - {os.path.join(output_dir, f'cluster_dist_capped_at_{cluster_size_cap}.png')}")
    print(f"  - {os.path.join(output_dir, 'kinetic_parameter_overlap.png')}")
    
    # Check if large clusters report exists
    large_clusters_report = os.path.join(output_dir, 'large_clusters_report.csv')
    if os.path.exists(large_clusters_report):
        print(f"  - {large_clusters_report}")
    
    # Report cross-validation folds if created
    if fold_info:
        cv_type = "Balanced" if use_balanced_cv and not compare_cv_methods else "Original"
        print(f"\n{cv_type} Cross-Validation Folds ({n_folds}-fold):")
        for fold_name, info in fold_info.items():
            print(f"  - {info['train_path']} ({info['train_count']} rows)")
            print(f"  - {info['valid_path']} ({info['valid_count']} rows)")
            
        if compare_cv_methods and use_balanced_cv:
            print(f"\nAdditional Balanced CV files in: cv_folds_balanced/")
    
    print("------------------------\n")
    
