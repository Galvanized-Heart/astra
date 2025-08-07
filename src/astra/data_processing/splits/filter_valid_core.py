"""
Module for filtering protein sequences based on core dataset membership.

This module provides functions to:
1. Extract core protein sequences from datasets
2. Filter validation fold files to keep only core sequences
3. Save filtered results to specified output directories
"""

import pandas as pd
from pathlib import Path
import os
from typing import Set, List, Union, Dict


def load_protein_sequences(file_path: Union[str, Path], sequence_column: str = 'protein_sequence') -> Set[str]:
    """
    Load protein sequences from a CSV file into a set for fast lookup.
    
    Args:
        file_path: Path to the CSV file
        sequence_column: Name of the column containing protein sequences
        
    Returns:
        Set of unique protein sequences
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If the sequence column doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        if sequence_column not in df.columns:
            raise KeyError(f"Column '{sequence_column}' not found in {file_path}")
        
        # Remove any NaN values and convert to set for O(1) lookup
        sequences = set(df[sequence_column].dropna().astype(str))
        print(f"Loaded {len(sequences)} unique sequences from {file_path}")
        return sequences
        
    except Exception as e:
        raise Exception(f"Error reading {file_path}: {str(e)}")


def get_core_sequences(core_file: Union[str, Path], 
                      pangenomic_file: Union[str, Path], 
                      sequence_column: str = 'protein_sequence') -> Set[str]:
    """
    Extract core sequences by comparing core and pangenomic datasets.
    
    Args:
        core_file: Path to core dataset CSV
        pangenomic_file: Path to pangenomic dataset CSV
        sequence_column: Name of the column containing protein sequences
        
    Returns:
        Set of protein sequences based on return_type
    """
    print("Loading core sequences...")
    core_sequences = load_protein_sequences(core_file, sequence_column)
    
    print("Loading pangenomic sequences...")
    pangenomic_sequences = load_protein_sequences(pangenomic_file, sequence_column)
    
    core_result = core_sequences
    print(f"Found {len(core_result)} core sequences")
    pangenomic_result = pangenomic_sequences - core_sequences
    print(f"Found {len(pangenomic_result)} sequences in pangenomic but not in core")

    return core_result


def filter_fold_by_sequences(fold_file: Union[str, Path], 
                           target_sequences: Set[str], 
                           sequence_column: str = 'protein_sequence',
                           keep_target: bool = True) -> pd.DataFrame:
    """
    Filter a fold CSV file to keep or remove sequences based on target set.
    
    Args:
        fold_file: Path to the fold CSV file
        target_sequences: Set of target protein sequences
        sequence_column: Name of the column containing protein sequences
        keep_target: If True, keep sequences in target_sequences; if False, remove them
        
    Returns:
        Filtered DataFrame
    """
    fold_file = Path(fold_file)
    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_file}")
    
    try:
        df = pd.read_csv(fold_file)
        if sequence_column not in df.columns:
            raise KeyError(f"Column '{sequence_column}' not found in {fold_file}")
        
        original_count = len(df)
        
        if keep_target:
            # Keep only sequences that are in target_sequences
            mask = df[sequence_column].astype(str).isin(target_sequences)
        else:
            # Remove sequences that are in target_sequences
            mask = ~df[sequence_column].astype(str).isin(target_sequences)
        
        filtered_df = df[mask].copy()
        filtered_count = len(filtered_df)
        
        print(f"Filtered {fold_file.name}: {original_count} -> {filtered_count} rows "
              f"({filtered_count/original_count*100:.1f}% retained)")
        
        return filtered_df
        
    except Exception as e:
        raise Exception(f"Error filtering {fold_file}: {str(e)}")


def process_cv_folds(fold_dir: Union[str, Path], 
                    target_sequences: Set[str],
                    output_dir: Union[str, Path],
                    fold_pattern: str = "fold_{}_valid.csv",
                    num_folds: int = 5,
                    sequence_column: str = 'protein_sequence',
                    keep_target: bool = True) -> Dict[str, int]:
    """
    Process multiple CV fold files and save filtered versions.
    
    Args:
        fold_dir: Directory containing the fold CSV files
        target_sequences: Set of target protein sequences for filtering
        output_dir: Directory to save filtered files
        fold_pattern: Pattern for fold file names (with {} for fold number)
        num_folds: Number of folds to process
        sequence_column: Name of the column containing protein sequences
        keep_target: If True, keep sequences in target_sequences; if False, remove them
        
    Returns:
        Dictionary mapping fold names to number of rows in filtered data
    """
    fold_dir = Path(fold_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for fold_idx in range(num_folds):
        fold_filename = fold_pattern.format(fold_idx)
        fold_path = fold_dir / fold_filename
        output_path = output_dir / fold_filename
        
        print(f"\nProcessing {fold_filename}...")
        
        try:
            filtered_df = filter_fold_by_sequences(
                fold_path, 
                target_sequences, 
                sequence_column, 
                keep_target
            )
            
            # Save filtered dataframe
            filtered_df.to_csv(output_path, index=False)
            print(f"Saved filtered data to {output_path}")
            
            results[fold_filename] = len(filtered_df)
            
        except Exception as e:
            print(f"Error processing {fold_filename}: {str(e)}")
            results[fold_filename] = 0
    
    return results


def filter_valid_core_pipeline(data_dir: Union[str, Path],
                              cv_dir: Union[str, Path],
                              output_dir: Union[str, Path],
                              core_filename: str = 'CPI_all_brenda_core_enriched.csv',
                              pangenomic_filename: str = 'CPI_all_brenda_pangenomic_enriched.csv',
                              fold_pattern: str = "fold_{}_valid.csv",
                              num_folds: int = 5,
                              sequence_column: str = 'protein_sequence') -> Dict[str, int]:
    """
    Complete pipeline to filter CV folds based on core sequences.
    
    Args:
        data_dir: Directory containing the core and pangenomic CSV files
        cv_dir: Directory containing the CV fold files
        output_dir: Directory to save filtered fold files
        core_filename: Name of the core dataset file
        pangenomic_filename: Name of the pangenomic dataset file
        fold_pattern: Pattern for fold file names
        num_folds: Number of folds to process
        sequence_column: Name of the column containing protein sequences
        
    Returns:
        Dictionary mapping fold names to number of rows in filtered data
    """
    print("=== Starting Core Sequence Filtering Pipeline ===\n")
    
    # Setup paths
    data_dir = Path(data_dir)
    core_file = data_dir / core_filename
    pangenomic_file = data_dir / pangenomic_filename
    
    # Step 1: Get core sequences
    print("Step 1: Extracting core sequences...")
    core_sequences = get_core_sequences(
        core_file, 
        pangenomic_file, 
        sequence_column
    )
    
    # Step 2: Filter CV folds
    print(f"\nStep 2: Filtering {num_folds} CV folds...")
    results = process_cv_folds(
        cv_dir,
        core_sequences,
        output_dir,
        fold_pattern,
        num_folds,
        sequence_column,
        keep_target=True  # Keep only core sequences
    )
    
    # Summary
    print(f"\n=== Pipeline Complete ===")
    print(f"Core sequences identified: {len(core_sequences)}")
    print(f"Filtered files saved to: {output_dir}")
    
    total_filtered_rows = sum(results.values())
    print(f"Total rows in filtered datasets: {total_filtered_rows}")
    
    return results