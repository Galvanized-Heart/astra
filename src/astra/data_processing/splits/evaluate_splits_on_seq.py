"""
Evaluation functions that ensures there is no data leakage of sequences 
in k-fold cross-validation splits based on protein sequences.

This script:
1. Loads k-fold CV split files from a specified directory
2. Checks for protein_sequence leakage within folds (train vs valid)
3. Checks validation partition integrity across folds for protein sequences
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
    Check for protein_sequence leakage within each fold (train vs validation).
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        Dictionary mapping fold_id to list of overlapping protein sequences
    """
    intra_fold_leaks = {}
    
    for fold_id, fold_data in cv_data.items():
        train_df = fold_data['train']
        valid_df = fold_data['valid']
        
        # Check if protein_sequence column exists
        if 'protein_sequence' not in train_df.columns or 'protein_sequence' not in valid_df.columns:
            logger.warning(f"Fold {fold_id}: Missing 'protein_sequence' column - skipping leakage check")
            continue
        
        # Get protein_sequences from each set
        train_sequences = set(train_df['protein_sequence'].unique())
        valid_sequences = set(valid_df['protein_sequence'].unique())
        
        # Find overlaps
        overlapping_sequences = train_sequences.intersection(valid_sequences)
        
        if overlapping_sequences:
            # Store first 50 characters of each sequence for readability
            overlapping_sequences_preview = [seq[:50] + "..." if len(seq) > 50 else seq 
                                           for seq in sorted(list(overlapping_sequences))]
            intra_fold_leaks[fold_id] = overlapping_sequences_preview
            logger.warning(f"Fold {fold_id}: {len(overlapping_sequences)} overlapping protein sequences between train/valid")
        else:
            logger.info(f"Fold {fold_id}: No intra-fold leakage detected")
            
    return intra_fold_leaks


def check_validation_partition(cv_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[str]]:
    """
    Check that validation sets are properly partitioned (disjoint) across folds.
    For protein sequence-based splits, it's expected that training sequences don't appear in validation.
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        Dictionary with validation partition issues
    """
    validation_issues = {}
    
    # Collect all sequences from validation sets
    valid_sequences_by_fold = {}
    
    for fold_id, fold_data in cv_data.items():
        if 'protein_sequence' not in fold_data['valid'].columns:
            logger.warning(f"Fold {fold_id}: Missing 'protein_sequence' column - skipping validation partition check")
            continue
            
        valid_sequences = set(fold_data['valid']['protein_sequence'].unique())
        valid_sequences_by_fold[fold_id] = valid_sequences
    
    # Check: Each validation sequence appears in exactly one fold (disjoint partition)
    sequence_valid_appearances = defaultdict(list)
    for fold_id, valid_sequences in valid_sequences_by_fold.items():
        for sequence in valid_sequences:
            sequence_valid_appearances[sequence].append(fold_id)
    
    multiple_appearances = []
    
    for sequence, appearances in sequence_valid_appearances.items():
        if len(appearances) > 1:
            # Store preview of sequence for readability
            seq_preview = sequence[:50] + "..." if len(sequence) > 50 else sequence
            multiple_appearances.append((seq_preview, appearances))
    
    # Report issues
    if multiple_appearances:
        validation_issues['multiple_appearances'] = multiple_appearances
        logger.warning(f"Found {len(multiple_appearances)} protein sequences appearing in multiple validation sets")
    
    if not validation_issues:
        logger.info("Validation partition is properly constructed - all validation sequences are disjoint across folds")
    
    return validation_issues


def check_train_valid_separation(cv_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, int]:
    """
    Check that training and validation sequences are properly separated across all folds.
    This is expected behavior for protein sequence-based splits.
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        Dictionary with separation statistics
    """
    separation_stats = {
        'total_train_sequences': 0,
        'total_valid_sequences': 0,
        'overlapping_sequences': 0,
        'separation_ratio': 0.0
    }
    
    # Collect all training and validation sequences
    all_train_sequences = set()
    all_valid_sequences = set()
    
    for fold_id, fold_data in cv_data.items():
        if 'protein_sequence' in fold_data['train'].columns:
            train_sequences = set(fold_data['train']['protein_sequence'].unique())
            all_train_sequences.update(train_sequences)
            
        if 'protein_sequence' in fold_data['valid'].columns:
            valid_sequences = set(fold_data['valid']['protein_sequence'].unique())
            all_valid_sequences.update(valid_sequences)
    
    # Calculate overlap
    overlapping_sequences = all_train_sequences.intersection(all_valid_sequences)
    
    separation_stats['total_train_sequences'] = len(all_train_sequences)
    separation_stats['total_valid_sequences'] = len(all_valid_sequences)
    separation_stats['overlapping_sequences'] = len(overlapping_sequences)
    
    total_sequences = len(all_train_sequences.union(all_valid_sequences))
    if total_sequences > 0:
        separation_stats['separation_ratio'] = (total_sequences - len(overlapping_sequences)) / total_sequences
    
    if len(overlapping_sequences) == 0:
        logger.info("Perfect train/validation separation achieved - no sequences overlap between training and validation sets")
    else:
        logger.info(f"Train/validation separation: {len(overlapping_sequences)} overlapping sequences out of {total_sequences} total")
    
    return separation_stats


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
            # Create preview versions for readability
            duplicate_pairs_preview = []
            for protein_seq, ligand_smiles in sorted(list(duplicate_pairs)):
                protein_preview = protein_seq[:50] + "..." if len(protein_seq) > 50 else protein_seq
                duplicate_pairs_preview.append((protein_preview, ligand_smiles))
            
            pair_duplicates[fold_id] = duplicate_pairs_preview
            logger.warning(f"Fold {fold_id}: {len(duplicate_pairs)} duplicate (protein_sequence, ligand_smiles) pairs between train/valid")
        else:
            logger.info(f"Fold {fold_id}: No duplicate pairs detected")
    
    return pair_duplicates


def analyze_sequence_distribution(cv_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze the distribution of protein_sequences across all folds and splits.
    
    Args:
        cv_data: Dictionary containing fold data
        
    Returns:
        DataFrame with sequence distribution analysis
    """
    sequence_info = defaultdict(lambda: defaultdict(int))
    
    for fold_id, fold_data in cv_data.items():
        # Count sequences in train set
        if 'protein_sequence' in fold_data['train'].columns:
            train_sequences = fold_data['train']['protein_sequence'].value_counts()
            for sequence, count in train_sequences.items():
                sequence_info[sequence][f'fold_{fold_id}_train'] = count
                
        # Count sequences in valid set
        if 'protein_sequence' in fold_data['valid'].columns:
            valid_sequences = fold_data['valid']['protein_sequence'].value_counts()
            for sequence, count in valid_sequences.items():
                sequence_info[sequence][f'fold_{fold_id}_valid'] = count
    
    # Convert to DataFrame
    sequence_df = pd.DataFrame.from_dict(sequence_info, orient='index').fillna(0).astype(int)
    
    # Add summary columns
    sequence_df['total_samples'] = sequence_df.sum(axis=1)
    sequence_df['num_folds_present'] = (sequence_df > 0).sum(axis=1)
    
    # Sort by total samples
    sequence_df = sequence_df.sort_values('total_samples', ascending=False)
    
    return sequence_df


def generate_summary_report(cv_data: Dict[str, Dict[str, pd.DataFrame]], 
                          intra_fold_leaks: Dict[str, List[str]], 
                          validation_issues: Dict[str, List],
                          pair_duplicates: Dict[str, List[Tuple[str, str]]],
                          sequence_df: pd.DataFrame,
                          separation_stats: Dict[str, int]) -> str:
    """
    Generate a comprehensive summary report.
    
    Returns:
        String containing the formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("K-FOLD CROSS-VALIDATION SPLITS EVALUATION REPORT (PROTEIN SEQUENCES)")
    report.append("=" * 80)
    report.append("")
    
    # Basic statistics
    report.append("DATASET OVERVIEW:")
    report.append("-" * 40)
    total_samples = 0
    total_sequences = len(sequence_df)
    
    for fold_id, fold_data in cv_data.items():
        train_size = len(fold_data['train'])
        valid_size = len(fold_data['valid'])
        total_samples += train_size + valid_size
        report.append(f"Fold {fold_id}: {train_size:,} train + {valid_size:,} valid = {train_size + valid_size:,} total")
    
    report.append(f"\nTotal samples across all folds: {total_samples:,}")
    report.append(f"Total unique protein sequences: {total_sequences:,}")
    report.append("")
    
    # Intra-fold leakage results
    report.append("INTRA-FOLD LEAKAGE ANALYSIS:")
    report.append("-" * 40)
    if not intra_fold_leaks:
        report.append("NO INTRA-FOLD LEAKAGE DETECTED")
        report.append("All protein sequences are properly separated between train/valid within each fold.")
    else:
        report.append("INTRA-FOLD LEAKAGE DETECTED")
        for fold_id, leaked_sequences in intra_fold_leaks.items():
            report.append(f"Fold {fold_id}: {len(leaked_sequences)} overlapping protein sequences")
            if len(leaked_sequences) <= 5:
                for seq in leaked_sequences:
                    report.append(f"  {seq}")
            else:
                for seq in leaked_sequences[:5]:
                    report.append(f"  {seq}")
                report.append(f"  ... and {len(leaked_sequences) - 5} more sequences")
    report.append("")
    
    # Validation partition analysis
    report.append("VALIDATION PARTITION ANALYSIS:")
    report.append("-" * 40)
    if not validation_issues:
        report.append("VALIDATION PARTITION IS PROPERLY CONSTRUCTED")
        report.append("All validation sequences are disjoint across folds (no sequence appears in multiple validation sets).")
        report.append("Training sequences properly excluded from validation sets (expected behavior).")
    else:
        report.append("VALIDATION PARTITION ISSUES DETECTED")
        
        if 'multiple_appearances' in validation_issues:
            multiple = validation_issues['multiple_appearances']
            report.append(f"Protein sequences appearing in multiple validation sets: {len(multiple)}")
            for seq_preview, folds in multiple[:3]:
                report.append(f"  {seq_preview}: appears in folds {folds}")
            if len(multiple) > 3:
                report.append(f"  ... and {len(multiple) - 3} more sequences")
    report.append("")
    
    # Train/Validation separation analysis
    report.append("TRAIN/VALIDATION SEPARATION ANALYSIS:")
    report.append("-" * 40)
    total_train = separation_stats['total_train_sequences']
    total_valid = separation_stats['total_valid_sequences']
    overlapping = separation_stats['overlapping_sequences']
    separation_ratio = separation_stats['separation_ratio']
    
    report.append(f"Total unique training sequences: {total_train:,}")
    report.append(f"Total unique validation sequences: {total_valid:,}")
    report.append(f"Overlapping sequences (train ∩ valid): {overlapping:,}")
    
    # Pair duplicates analysis
    report.append("\nPAIR DUPLICATES ANALYSIS:")
    report.append("-" * 40)
    if not pair_duplicates:
        report.append("NO DUPLICATE PAIRS DETECTED")
        report.append("No (protein_sequence, ligand_smiles) pairs appear in both train/valid within any fold.")
    else:
        report.append("DUPLICATE PAIRS DETECTED")
        for fold_id, duplicate_pairs in pair_duplicates.items():
            report.append(f"Fold {fold_id}: {len(duplicate_pairs)} duplicate pairs")
            if len(duplicate_pairs) <= 3:
                for pair in duplicate_pairs:
                    report.append(f"  ({pair[0]}, {pair[1]})")
            else:
                report.append(f"  ... showing first 3 pairs")
                for pair in duplicate_pairs[:3]:
                    report.append(f"  ({pair[0]}, {pair[1]})")
    report.append("")
    
    # Sequence distribution insights
    report.append("PROTEIN SEQUENCE DISTRIBUTION:")
    report.append("-" * 40)
    multi_fold_sequences = sequence_df[sequence_df['num_folds_present'] > 1]
    if len(multi_fold_sequences) > 0:
        report.append(f"{len(multi_fold_sequences)} protein sequences appear in multiple folds")
    else:
        report.append("All protein sequences appear in exactly one fold")
    
    report.append("")
    
    # Overall assessment
    report.append("OVERALL ASSESSMENT:")
    report.append("-" * 40)
    
    # Count actual issues (intra-fold leaks and validation partition issues are problems)
    # Train/validation separation is expected and good behavior
    has_issues = bool(intra_fold_leaks or validation_issues or pair_duplicates)
    
    if not has_issues:
        report.append("VALIDATION PASSED")
        report.append("The k-fold splits are properly constructed for protein sequence-based evaluation:")
        report.append("No intra-fold leakage (train/valid within same fold are separate)")
        report.append("No duplicate pairs within folds")
        if separation_stats['overlapping_sequences'] == 0:
            report.append("Perfect train/validation separation (expected for protein sequences)")
    else:
        report.append("VALIDATION FAILED")
        report.append("Issues detected that may compromise model evaluation:")
        if intra_fold_leaks:
            report.append("Intra-fold leakage detected")
        if validation_issues:
            report.append("Validation partition issues detected")
        if pair_duplicates:
            report.append("Duplicate pairs detected")
        report.append("Review the splits before using for model training.")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def plot_valid_valid_heatmap(cv_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str = None) -> plt.Figure:
    """
    Plot a validation-validation overlap heatmap for protein sequences. Should be diagonal (no overlap between validation sets).
    
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
    
    # Get validation sequences for each fold
    valid_sequences = {}
    for i, fold_id in enumerate(fold_ids):
        if 'protein_sequence' in cv_data[fold_id]['valid'].columns:
            valid_sequences[i] = set(cv_data[fold_id]['valid']['protein_sequence'].unique())
        else:
            valid_sequences[i] = set()
    
    # Calculate overlaps
    for i in range(n_folds):
        for j in range(n_folds):
            if i == j:
                overlap_matrix[i, j] = len(valid_sequences[i])  # Self-overlap is the size
            else:
                overlap_matrix[i, j] = len(valid_sequences[i].intersection(valid_sequences[j]))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use different color for diagonal vs off-diagonal
    mask = np.zeros_like(overlap_matrix, dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    
    # Plot off-diagonal (should be zeros)
    sns.heatmap(overlap_matrix, mask=mask, annot=True, fmt='.0f', cmap='Reds', 
                cbar_kws={'label': 'Protein Sequence Overlap Count'}, ax=ax, vmin=0)
    
    # Plot diagonal with different color
    sns.heatmap(overlap_matrix, mask=~mask, annot=True, fmt='.0f', cmap='Blues', 
                cbar=False, ax=ax, alpha=0.7)
    
    # Customize plot
    ax.set_title('Validation Set Protein Sequence Overlaps\n(Diagonal = Validation Set Size, Off-diagonal Should be 0)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold Index', fontsize=12)
    ax.set_ylabel('Fold Index', fontsize=12)
    ax.set_xticklabels([f'Fold {fold_id}' for fold_id in fold_ids], rotation=45)
    ax.set_yticklabels([f'Fold {fold_id}' for fold_id in fold_ids], rotation=0)
    
    plt.tight_layout()
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'valid_valid_heatmap_seq.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Valid-valid sequence heatmap saved to {output_dir}/valid_valid_heatmap_seq.png")
    
    return fig


def plot_valid_train_heatmap(cv_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str = None) -> plt.Figure:
    """
    Plot a validation-train overlap heatmap for protein sequences. Row i should be 0 at (i,i) and constant |valid_i| elsewhere.
    
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
    
    # Get sequences for each fold
    valid_sequences = {}
    train_sequences = {}
    for i, fold_id in enumerate(fold_ids):
        if 'protein_sequence' in cv_data[fold_id]['valid'].columns:
            valid_sequences[i] = set(cv_data[fold_id]['valid']['protein_sequence'].unique())
        else:
            valid_sequences[i] = set()
            
        if 'protein_sequence' in cv_data[fold_id]['train'].columns:
            train_sequences[i] = set(cv_data[fold_id]['train']['protein_sequence'].unique())
        else:
            train_sequences[i] = set()
    
    # Calculate overlaps: validation_i vs train_j
    for i in range(n_folds):  # validation fold
        for j in range(n_folds):  # train fold
            overlap_matrix[i, j] = len(valid_sequences[i].intersection(train_sequences[j]))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap that highlights zeros differently
    from matplotlib.colors import ListedColormap
    colors = ['white'] + list(plt.cm.Reds(np.linspace(0.2, 1, 256)))
    custom_cmap = ListedColormap(colors)
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap=custom_cmap, 
                cbar_kws={'label': 'Protein Sequence Overlap Count'}, ax=ax, vmin=0)
    
    # Highlight diagonal (should be zeros)
    for i in range(n_folds):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))
    
    # Customize plot
    ax.set_title('Validation vs Train Protein Sequence Overlaps\n(Diagonal Should be 0, Rows Should be Constant)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Train Fold Index', fontsize=12)
    ax.set_ylabel('Validation Fold Index', fontsize=12)
    ax.set_xticklabels([f'Train {fold_id}' for fold_id in fold_ids], rotation=45)
    ax.set_yticklabels([f'Valid {fold_id}' for fold_id in fold_ids], rotation=0)
    
    plt.tight_layout()
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'valid_train_heatmap_seq.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Valid-train sequence heatmap saved to {output_dir}/valid_train_heatmap_seq.png")
    
    return fig


def plot_sequence_length_distribution(cv_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str = None) -> plt.Figure:
    """
    Plot histogram/boxplot of protein sequence lengths per fold.
    
    Args:
        cv_data: Dictionary containing fold data
        output_dir: Optional directory to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fold_ids = sorted(cv_data.keys())
    
    # Collect sequence lengths for each fold
    fold_sequence_data = {}
    all_sequence_lengths = []
    
    for fold_id in fold_ids:
        # Combine train and valid data
        combined_df = pd.concat([cv_data[fold_id]['train'], cv_data[fold_id]['valid']], ignore_index=True)
        
        if 'protein_sequence' in combined_df.columns:
            sequence_lengths = combined_df['protein_sequence'].str.len().values
            fold_sequence_data[fold_id] = sequence_lengths
            all_sequence_lengths.extend(sequence_lengths)
        else:
            fold_sequence_data[fold_id] = []
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 12))
    
    # 1. Overall sequence length distribution (histogram)
    if all_sequence_lengths:
        ax1.hist(all_sequence_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(all_sequence_lengths), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(all_sequence_lengths):.1f}')
        ax1.axvline(np.median(all_sequence_lengths), color='orange', linestyle='--', 
                    label=f'Median: {np.median(all_sequence_lengths):.1f}')
        ax1.legend()
    
    ax1.set_xlabel('Protein Sequence Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Protein Sequence Length Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot of sequence lengths per fold
    sequence_lengths_list = [fold_sequence_data[fold_id] for fold_id in fold_ids if len(fold_sequence_data[fold_id]) > 0]
    valid_fold_ids = [fold_id for fold_id in fold_ids if len(fold_sequence_data[fold_id]) > 0]
    
    if sequence_lengths_list:
        box_plot = ax2.boxplot(sequence_lengths_list, labels=[f'Fold {fid}' for fid in valid_fold_ids], patch_artist=True)
        
        # Color all boxes the same
        for patch in box_plot['boxes']:
            patch.set_facecolor('lightblue')
    
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Protein Sequence Length')
    ax2.set_title('Protein Sequence Length Distribution per Fold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Sequence length distribution plot saved to {output_dir}/sequence_length_distribution.png")
    
    return fig


def evaluate_cv_splits_on_seq(data_dir: str, output_file: str = None, plot_dir: str = None, 
                             generate_plots: bool = False) -> Tuple[bool, str]:
    """
    Main function to evaluate k-fold CV splits for data leakage and partition integrity based on protein sequences.
    
    Args:
        data_dir: Path to directory containing fold CSV files
        output_file: Optional path to save the report
        plot_dir: Optional directory to save plots
        generate_plots: Whether to generate visualization plots
        
    Returns:
        Tuple of (validation_passed: bool, report: str)
    """
    try:
        logger.info(f"Starting evaluation of CV splits (protein sequences) in: {data_dir}")
        
        # Load data
        cv_data = load_cv_splits(data_dir)
        if not cv_data:
            raise ValueError("No valid fold data loaded")
        
        # Check for leakage and issues
        intra_fold_leaks = check_intra_fold_leakage(cv_data)
        validation_issues = check_validation_partition(cv_data)
        pair_duplicates = check_pair_duplicates(cv_data)
        
        # Check train/validation separation (expected behavior)
        separation_stats = check_train_valid_separation(cv_data)
        
        # Analyze sequence distribution
        sequence_df = analyze_sequence_distribution(cv_data)
        
        # Generate plots if requested
        if generate_plots:
            logger.info("Generating visualization plots...")
            try:
                plot_valid_valid_heatmap(cv_data, plot_dir)
                plot_valid_train_heatmap(cv_data, plot_dir)
                plot_sequence_length_distribution(cv_data, plot_dir)
                logger.info("All plots generated successfully")
            except Exception as e:
                logger.warning(f"Error generating plots: {e}")
        
        # Generate report
        report = generate_summary_report(cv_data, intra_fold_leaks, validation_issues, 
                                       pair_duplicates, sequence_df, separation_stats)
        
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
    
    parser = argparse.ArgumentParser(description="Evaluate k-fold CV splits for data leakage and partition integrity based on protein sequences")
    parser.add_argument("data_dir", help="Directory containing fold CSV files")
    parser.add_argument("--output", "-o", help="Output file for the report")
    parser.add_argument("--plot-dir", "-p", help="Directory to save visualization plots")
    parser.add_argument("--plots", action="store_true", help="Generate visualization plots")
    
    args = parser.parse_args()
    
    validation_passed, report = evaluate_cv_splits_on_seq(
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