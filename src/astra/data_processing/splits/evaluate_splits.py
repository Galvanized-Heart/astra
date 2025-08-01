"""
Evaluation functions that ensures there is no data leakage of sequences 
in k-fold cross-validation splits.

This script:
1. Loads k-fold CV split files from a specified directory
2. Checks for cluster_id leakage within folds (train vs valid)
3. Checks validation partition integrity across folds
4. Checks for duplicate (protein_sequence, ligand_smiles) pairs within folds
5. Provides detailed reports on data integrity
"""

import os
import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cv_splits(data_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all k-fold CV split files from a directory.
    
    Args:
        data_dir: Path to directory containing fold CSV files
        
    Returns:
        Dictionary with structure: {fold_id: {'train': df, 'valid': df}}
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    # Find all fold files
    train_files = glob.glob(str(data_dir / "fold_*_train.csv"))
    valid_files = glob.glob(str(data_dir / "fold_*_valid.csv"))
    
    if not train_files or not valid_files:
        raise ValueError(f"No fold files found in {data_dir}")
    
    # Extract fold numbers
    fold_numbers = set()
    for file in train_files:
        fold_num = os.path.basename(file).split('_')[1]
        fold_numbers.add(fold_num)
    
    for file in valid_files:
        fold_num = os.path.basename(file).split('_')[1]
        fold_numbers.add(fold_num)
    
    fold_numbers = sorted(fold_numbers)
    logger.info(f"Found {len(fold_numbers)} folds: {fold_numbers}")
    
    # Load data
    cv_data = {}
    for fold_num in fold_numbers:
        train_path = data_dir / f"fold_{fold_num}_train.csv"
        valid_path = data_dir / f"fold_{fold_num}_valid.csv"
        
        if not train_path.exists() or not valid_path.exists():
            logger.warning(f"Missing files for fold {fold_num}")
            continue
            
        try:
            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path)
            
            cv_data[fold_num] = {
                'train': train_df,
                'valid': valid_df
            }
            
            logger.info(f"Loaded fold {fold_num}: {len(train_df)} train, {len(valid_df)} valid samples")
            
        except Exception as e:
            logger.error(f"Error loading fold {fold_num}: {e}")
            
    return cv_data


def check_intra_fold_leakage(cv_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[str]]:
    """
    Check for cluster_id leakage within each fold (train vs validation).
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        Dictionary mapping fold_id to list of overlapping cluster_ids
    """
    intra_fold_leaks = {}
    
    for fold_id, fold_data in cv_data.items():
        train_df = fold_data['train']
        valid_df = fold_data['valid']
        
        # Get cluster_ids from each set
        train_clusters = set(train_df['cluster_id'].unique())
        valid_clusters = set(valid_df['cluster_id'].unique())
        
        # Find overlaps
        overlapping_clusters = train_clusters.intersection(valid_clusters)
        
        if overlapping_clusters:
            intra_fold_leaks[fold_id] = sorted(list(overlapping_clusters))
            logger.warning(f"Fold {fold_id}: {len(overlapping_clusters)} overlapping clusters between train/valid")
        else:
            logger.info(f"Fold {fold_id}: No intra-fold leakage detected")
            
    return intra_fold_leaks


def check_validation_partition(cv_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[str]]:
    """
    Check that each cluster_id appears in validation in exactly one fold (disjoint partition),
    and that the union of all validation clusters equals all clusters in the dataset.
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        Dictionary with validation partition issues
    """
    validation_issues = {}
    
    # Collect all clusters from validation sets
    all_valid_clusters = set()
    valid_clusters_by_fold = {}
    
    for fold_id, fold_data in cv_data.items():
        valid_clusters = set(fold_data['valid']['cluster_id'].unique())
        valid_clusters_by_fold[fold_id] = valid_clusters
        all_valid_clusters.update(valid_clusters)
    
    # Collect all clusters in the entire dataset
    all_dataset_clusters = set()
    for fold_id, fold_data in cv_data.items():
        train_clusters = set(fold_data['train']['cluster_id'].unique())
        valid_clusters = set(fold_data['valid']['cluster_id'].unique())
        all_dataset_clusters.update(train_clusters)
        all_dataset_clusters.update(valid_clusters)
    
    # Check 1: Each cluster appears in validation exactly once
    cluster_valid_appearances = defaultdict(list)
    for fold_id, valid_clusters in valid_clusters_by_fold.items():
        for cluster_id in valid_clusters:
            cluster_valid_appearances[cluster_id].append(fold_id)
    
    multiple_appearances = []
    missing_from_validation = []
    
    for cluster_id in all_dataset_clusters:
        appearances = cluster_valid_appearances.get(cluster_id, [])
        if len(appearances) > 1:
            multiple_appearances.append((cluster_id, appearances))
        elif len(appearances) == 0:
            missing_from_validation.append(cluster_id)
    
    # Check 2: Union of all validation clusters equals all dataset clusters
    missing_clusters = all_dataset_clusters - all_valid_clusters
    extra_clusters = all_valid_clusters - all_dataset_clusters
    
    # Print issues
    if multiple_appearances:
        validation_issues['multiple_appearances'] = multiple_appearances
        logger.warning(f"Found {len(multiple_appearances)} clusters appearing in multiple validation sets")
    
    if missing_from_validation:
        validation_issues['missing_from_validation'] = missing_from_validation
        logger.warning(f"Found {len(missing_from_validation)} clusters missing from all validation sets")
    
    if missing_clusters:
        validation_issues['missing_clusters'] = sorted(list(missing_clusters))
        logger.warning(f"Found {len(missing_clusters)} clusters in dataset but not in validation union")
    
    if extra_clusters:
        validation_issues['extra_clusters'] = sorted(list(extra_clusters))
        logger.warning(f"Found {len(extra_clusters)} clusters in validation union but not in dataset")
    
    if not validation_issues:
        logger.info("Validation partition is properly constructed")
    
    return validation_issues


def check_pair_duplicates(cv_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Check for duplicate (protein_sequence, ligand_smiles) pairs within the same fold
    between train and validation sets.
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        Dictionary mapping fold_id to list of duplicate pairs
    """
    pair_duplicates = {}
    
    for fold_id, fold_data in cv_data.items():
        train_df = fold_data['train']
        valid_df = fold_data['valid']
        
        # Check if required columns exist
        required_cols = ['protein_sequence', 'ligand_smiles']
        missing_cols = [col for col in required_cols if col not in train_df.columns or col not in valid_df.columns]
        
        if missing_cols:
            logger.warning(f"Fold {fold_id}: Missing columns {missing_cols} - skipping pair duplicate check")
            continue
        
        # Create sets of (protein_sequence, ligand_smiles) pairs
        train_pairs = set(zip(train_df['protein_sequence'], train_df['ligand_smiles']))
        valid_pairs = set(zip(valid_df['protein_sequence'], valid_df['ligand_smiles']))
        
        # Find duplicates
        duplicate_pairs = train_pairs.intersection(valid_pairs)
        
        if duplicate_pairs:
            pair_duplicates[fold_id] = sorted(list(duplicate_pairs))
            logger.warning(f"Fold {fold_id}: {len(duplicate_pairs)} duplicate (protein_sequence, ligand_smiles) pairs between train/valid")
        else:
            logger.info(f"Fold {fold_id}: No duplicate pairs detected")
    
    return pair_duplicates


def analyze_cluster_distribution(cv_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze the distribution of cluster_ids across all folds and splits.
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        DataFrame with cluster distribution analysis
    """
    cluster_info = defaultdict(lambda: defaultdict(int))
    
    for fold_id, fold_data in cv_data.items():
        # Count clusters in train set
        train_clusters = fold_data['train']['cluster_id'].value_counts()
        for cluster_id, count in train_clusters.items():
            cluster_info[cluster_id][f'fold_{fold_id}_train'] = count
            
        # Count clusters in valid set
        valid_clusters = fold_data['valid']['cluster_id'].value_counts()
        for cluster_id, count in valid_clusters.items():
            cluster_info[cluster_id][f'fold_{fold_id}_valid'] = count
    
    # Convert to DataFrame
    cluster_df = pd.DataFrame.from_dict(cluster_info, orient='index').fillna(0).astype(int)
    
    # Add summary columns
    cluster_df['total_samples'] = cluster_df.sum(axis=1)
    cluster_df['num_folds_present'] = (cluster_df > 0).sum(axis=1)
    
    # Sort by total samples
    cluster_df = cluster_df.sort_values('total_samples', ascending=False)
    
    return cluster_df

def generate_summary_report(cv_data: Dict[str, Dict[str, pd.DataFrame]], 
                          intra_fold_leaks: Dict[str, List[str]], 
                          validation_issues: Dict[str, List],
                          pair_duplicates: Dict[str, List[Tuple[str, str]]],
                          cluster_df: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary report.
    
    Returns:
        String containing the formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("K-FOLD CROSS-VALIDATION SPLITS EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Basic statistics
    report.append("DATASET OVERVIEW:")
    report.append("-" * 40)
    total_samples = 0
    total_clusters = len(cluster_df)
    
    for fold_id, fold_data in cv_data.items():
        train_size = len(fold_data['train'])
        valid_size = len(fold_data['valid'])
        total_samples += train_size + valid_size
        report.append(f"Fold {fold_id}: {train_size:,} train + {valid_size:,} valid = {train_size + valid_size:,} total")
    
    report.append(f"\nTotal samples across all folds: {total_samples:,}")
    report.append(f"Total unique clusters: {total_clusters:,}")
    report.append("")
    
    # Intra-fold leakage results
    report.append("INTRA-FOLD LEAKAGE ANALYSIS:")
    report.append("-" * 40)
    if not intra_fold_leaks:
        report.append("NO INTRA-FOLD LEAKAGE DETECTED")
        report.append("All clusters are properly separated between train/valid within each fold.")
    else:
        report.append("INTRA-FOLD LEAKAGE DETECTED")
        for fold_id, leaked_clusters in intra_fold_leaks.items():
            report.append(f"Fold {fold_id}: {len(leaked_clusters)} overlapping clusters")
            if len(leaked_clusters) <= 10:
                report.append(f"  Clusters: {leaked_clusters}")
            else:
                report.append(f"  Clusters: {leaked_clusters[:10]}... (showing first 10)")
    report.append("")
    
    # Validation partition analysis
    report.append("VALIDATION PARTITION ANALYSIS:")
    report.append("-" * 40)
    if not validation_issues:
        report.append("VALIDATION PARTITION IS PROPERLY CONSTRUCTED")
        report.append("Each cluster appears in validation exactly once, and all clusters are covered.")
    else:
        report.append("VALIDATION PARTITION ISSUES DETECTED")
        
        if 'multiple_appearances' in validation_issues:
            multiple = validation_issues['multiple_appearances']
            report.append(f"Clusters appearing in multiple validation sets: {len(multiple)}")
            for cluster_id, folds in multiple[:5]:
                report.append(f"  Cluster {cluster_id}: appears in folds {folds}")
            if len(multiple) > 5:
                report.append(f"  ... and {len(multiple) - 5} more")
        
        if 'missing_from_validation' in validation_issues:
            missing = validation_issues['missing_from_validation']
            report.append(f"Clusters missing from all validation sets: {len(missing)}")
            if len(missing) <= 10:
                report.append(f"  Clusters: {missing}")
            else:
                report.append(f"  Clusters: {missing[:10]}... (showing first 10)")
    report.append("")
    
    # Pair duplicates analysis
    report.append("PAIR DUPLICATES ANALYSIS:")
    report.append("-" * 40)
    if not pair_duplicates:
        report.append("NO DUPLICATE PAIRS DETECTED")
        report.append("No (protein_sequence, substrate) pairs appear in both train/valid within any fold.")
    else:
        report.append("DUPLICATE PAIRS DETECTED")
        for fold_id, duplicate_pairs in pair_duplicates.items():
            report.append(f"Fold {fold_id}: {len(duplicate_pairs)} duplicate pairs")
            if len(duplicate_pairs) <= 3:
                for pair in duplicate_pairs:
                    protein_preview = pair[0][:50] + "..." if len(pair[0]) > 50 else pair[0]
                    report.append(f"  ({protein_preview}, {pair[1]})")
            else:
                report.append(f"  ... showing first 3 pairs")
                for pair in duplicate_pairs[:3]:
                    protein_preview = pair[0][:50] + "..." if len(pair[0]) > 50 else pair[0]
                    report.append(f"  ({protein_preview}, {pair[1]})")
    report.append("")
    
    # Cluster distribution insights
    report.append("CLUSTER DISTRIBUTION:")
    report.append("-" * 40)
    multi_fold_clusters = cluster_df[cluster_df['num_folds_present'] > 1]
    if len(multi_fold_clusters) > 0:
        report.append(f"{len(multi_fold_clusters)} clusters appear in multiple folds")
        report.append("Top clusters appearing in multiple folds:")
        for idx, (cluster_id, row) in enumerate(multi_fold_clusters.head(5).iterrows()):
            report.append(f"  Cluster {cluster_id}: appears in {row['num_folds_present']} folds "
                         f"({row['total_samples']} samples)")
    else:
        report.append("All clusters appear in exactly one fold")
    
    report.append("")
    
    # Overall assessment
    report.append("OVERALL ASSESSMENT:")
    report.append("-" * 40)
    if not intra_fold_leaks and not validation_issues and not pair_duplicates:
        report.append("VALIDATION PASSED")
        report.append("The k-fold splits are properly constructed with no data leakage.")
    else:
        report.append("VALIDATION FAILED")
        report.append("Data leakage or partition issues detected. Review the splits before using for model training.")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def plot_valid_valid_heatmap(cv_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str = None) -> plt.Figure:
    """
    Plot a validation-validation overlap heatmap. Should be diagonal (no overlap between validation sets).
    
    Args:
        cv_data: Dictionary containing fold data
        output_dir: Optional directory to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fold_ids = sorted(cv_data.keys())
    n_folds = len(fold_ids)
    
    # Create overlap matrix
    overlap_matrix = np.zeros((n_folds, n_folds))
    
    # Get validation clusters for each fold
    valid_clusters = {}
    for i, fold_id in enumerate(fold_ids):
        valid_clusters[i] = set(cv_data[fold_id]['valid']['cluster_id'].unique())
    
    # Calculate overlaps
    for i in range(n_folds):
        for j in range(n_folds):
            if i == j:
                overlap_matrix[i, j] = len(valid_clusters[i])  # Self-overlap is the size
            else:
                overlap_matrix[i, j] = len(valid_clusters[i].intersection(valid_clusters[j]))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use different color for diagonal vs off-diagonal
    mask = np.zeros_like(overlap_matrix, dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    
    # Plot off-diagonal (should be zeros)
    sns.heatmap(overlap_matrix, mask=mask, annot=True, fmt='.0f', cmap='Reds', 
                cbar_kws={'label': 'Cluster Overlap Count'}, ax=ax, vmin=0)
    
    # Plot diagonal with different color
    sns.heatmap(overlap_matrix, mask=~mask, annot=True, fmt='.0f', cmap='Blues', 
                cbar=False, ax=ax, alpha=0.7)
    
    # Customize plot
    ax.set_title('Validation Set Cluster Overlaps\n(Diagonal = Validation Set Size, Off-diagonal Should be 0)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold Index', fontsize=12)
    ax.set_ylabel('Fold Index', fontsize=12)
    ax.set_xticklabels([f'Fold {fold_id}' for fold_id in fold_ids], rotation=45)
    ax.set_yticklabels([f'Fold {fold_id}' for fold_id in fold_ids], rotation=0)
    
    plt.tight_layout()
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'valid_valid_heatmap.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Valid-valid heatmap saved to {output_dir}/valid_valid_heatmap.png")
    
    return fig


def plot_valid_train_heatmap(cv_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str = None) -> plt.Figure:
    """
    Plot a validation-train overlap heatmap. Row i should be 0 at (i,i) and constant |valid_i| elsewhere.
    
    Args:
        cv_data: Dictionary containing fold data
        output_dir: Optional directory to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fold_ids = sorted(cv_data.keys())
    n_folds = len(fold_ids)
    
    # Create overlap matrix
    overlap_matrix = np.zeros((n_folds, n_folds))
    
    # Get clusters for each fold
    valid_clusters = {}
    train_clusters = {}
    for i, fold_id in enumerate(fold_ids):
        valid_clusters[i] = set(cv_data[fold_id]['valid']['cluster_id'].unique())
        train_clusters[i] = set(cv_data[fold_id]['train']['cluster_id'].unique())
    
    # Calculate overlaps: validation_i vs train_j
    for i in range(n_folds):  # validation fold
        for j in range(n_folds):  # train fold
            overlap_matrix[i, j] = len(valid_clusters[i].intersection(train_clusters[j]))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap that highlights zeros differently
    from matplotlib.colors import ListedColormap
    colors = ['white'] + list(plt.cm.Reds(np.linspace(0.2, 1, 256)))
    custom_cmap = ListedColormap(colors)
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap=custom_cmap, 
                cbar_kws={'label': 'Cluster Overlap Count'}, ax=ax, vmin=0)
    
    # Highlight diagonal (should be zeros)
    for i in range(n_folds):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))
    
    # Customize plot
    ax.set_title('Validation vs Train Cluster Overlaps\n(Diagonal Should be 0, Rows Should be Constant)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Train Fold Index', fontsize=12)
    ax.set_ylabel('Validation Fold Index', fontsize=12)
    ax.set_xticklabels([f'Train {fold_id}' for fold_id in fold_ids], rotation=45)
    ax.set_yticklabels([f'Valid {fold_id}' for fold_id in fold_ids], rotation=0)
    
    plt.tight_layout()
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'valid_train_heatmap.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Valid-train heatmap saved to {output_dir}/valid_train_heatmap.png")
    
    return fig


def plot_cluster_size_distribution(cv_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str = None) -> plt.Figure:
    """
    Plot histogram/boxplot of cluster sizes per fold and flag folds dominated by very large clusters.
    
    Args:
        cv_data: Dictionary containing fold data
        output_dir: Optional directory to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fold_ids = sorted(cv_data.keys())
    
    # Collect cluster sizes for each fold
    fold_cluster_data = {}
    all_cluster_sizes = []
    
    for fold_id in fold_ids:
        # Combine train and valid data
        combined_df = pd.concat([cv_data[fold_id]['train'], cv_data[fold_id]['valid']], ignore_index=True)
        cluster_sizes = combined_df['cluster_id'].value_counts().values
        fold_cluster_data[fold_id] = cluster_sizes
        all_cluster_sizes.extend(cluster_sizes)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 12))
    
    # 1. Overall cluster size distribution (histogram)
    ax1.hist(all_cluster_sizes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Cluster Size')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Cluster Size Distribution')
    ax1.axvline(np.mean(all_cluster_sizes), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_cluster_sizes):.1f}')
    ax1.axvline(np.median(all_cluster_sizes), color='orange', linestyle='--', 
                label=f'Median: {np.median(all_cluster_sizes):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot of cluster sizes per fold
    cluster_sizes_list = [fold_cluster_data[fold_id] for fold_id in fold_ids]
    box_plot = ax2.boxplot(cluster_sizes_list, labels=[f'Fold {fid}' for fid in fold_ids], patch_artist=True)
    
    # Color boxes based on dominance
    colors = []
    dominant_folds = []
    
    for i, fold_id in enumerate(fold_ids):
        cluster_sizes = fold_cluster_data[fold_id]
        max_cluster = np.max(cluster_sizes)
        total_samples = np.sum(cluster_sizes)
        dominance_ratio = max_cluster / total_samples
        
        if dominance_ratio > 0.3:  # Flag if largest cluster is >30% of fold
            colors.append('lightcoral')
            dominant_folds.append((fold_id, dominance_ratio, max_cluster))
        elif dominance_ratio > 0.2:  # Warning if >20%
            colors.append('lightyellow')
        else:
            colors.append('lightblue')
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Cluster Size')
    ax2.set_title('Cluster Size Distribution per Fold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'cluster_size_distribution.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Cluster size distribution plot saved to {output_dir}/cluster_size_distribution.png")
    
    return fig


def evaluate_cv_splits(data_dir: str, output_file: str = None, plot_dir: str = None, 
                      generate_plots: bool = False) -> Tuple[bool, str]:
    """
    Main function to evaluate k-fold CV splits for data leakage and partition integrity.
    
    Args:
        data_dir: Path to directory containing fold CSV files
        output_file: Optional path to save the report
        plot_dir: Optional directory to save plots
        generate_plots: Whether to generate visualization plots
        
    Returns:
        Tuple of (validation_passed: bool, report: str)
    """
    try:
        logger.info(f"Starting evaluation of CV splits in: {data_dir}")
        
        # Load data
        cv_data = load_cv_splits(data_dir)
        if not cv_data:
            raise ValueError("No valid fold data loaded")
        
        # Check for leakage and issues
        intra_fold_leaks = check_intra_fold_leakage(cv_data)
        validation_issues = check_validation_partition(cv_data)
        pair_duplicates = check_pair_duplicates(cv_data)
        
        # Analyze cluster distribution
        cluster_df = analyze_cluster_distribution(cv_data)
        
        # Generate plots if requested
        if generate_plots:
            logger.info("Generating visualization plots...")
            try:
                plot_valid_valid_heatmap(cv_data, plot_dir)
                plot_valid_train_heatmap(cv_data, plot_dir)
                plot_cluster_size_distribution(cv_data, plot_dir)
                logger.info("All plots generated successfully")
            except Exception as e:
                logger.warning(f"Error generating plots: {e}")
        
        # Generate report
        report = generate_summary_report(cv_data, intra_fold_leaks, validation_issues, 
                                       pair_duplicates, cluster_df)
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_file}")
        
        # Print report
        print(report)
        
        # Return validation result
        validation_passed = not intra_fold_leaks and not validation_issues and not pair_duplicates
        
        return validation_passed, report
        
    except Exception as e:
        error_msg = f"Error during evaluation: {e}"
        logger.error(error_msg)
        return False, error_msg


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate k-fold CV splits for data leakage and partition integrity")
    parser.add_argument("data_dir", help="Directory containing fold CSV files")
    parser.add_argument("--output", "-o", help="Output file for the report")
    parser.add_argument("--plot-dir", "-p", help="Directory to save visualization plots")
    parser.add_argument("--plots", action="store_true", help="Generate visualization plots")
    
    args = parser.parse_args()
    
    validation_passed, report = evaluate_cv_splits(
        args.data_dir, 
        args.output, 
        args.plot_dir, 
        args.plots
    )
    
    if validation_passed:
        logger.info("Validation passed - no data leakage detected")
        exit(0)
    else:
        logger.error("Validation failed - data leakage detected")
        exit(1)