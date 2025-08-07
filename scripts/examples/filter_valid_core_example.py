"""
Example script to run the core sequence filtering pipeline.

This script demonstrates how to use the filter_valid_core module to:
1. Extract core protein sequences from datasets
2. Filter CV fold validation files to keep only core sequences
3. Save filtered results to a new directory
"""

import sys
import os
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]
ASTRA_DIR = ROOT_DIR / "src" / "astra"
DATA_DIR = ROOT_DIR / "data" / "processed"
CV_DIR = ROOT_DIR / "results" / "cv_splits_pangenomic" / "cv_folds_balanced"
OUTPUT_DIR = ROOT_DIR / "results" / "cv_splits_pangenomic" / "cv_folds_balanced" / "valid_core_filtered"

# Add ASTRA_DIR to Python path for imports
sys.path.append(str(ASTRA_DIR))

# Import the filtering functions
from astra.data_processing.splits.filter_valid_core import (
    filter_valid_core_pipeline,
    get_core_sequences,
    process_cv_folds,
    load_protein_sequences,
    filter_fold_by_sequences
)


def main():
    print(f"\n=== Running core sequence filtering pipeline in: {DATA_DIR} ===")

    # Run the complete pipeline
    results = filter_valid_core_pipeline(
        data_dir=DATA_DIR,
        cv_dir=CV_DIR,
        output_dir=OUTPUT_DIR,
        core_filename='CPI_all_brenda_core_enriched.csv',
        pangenomic_filename='CPI_all_brenda_pangenomic_enriched.csv',
        fold_pattern="fold_{}_valid.csv",
        num_folds=5,
        sequence_column='protein_sequence'
    )
    
    # Display results summary
    print("\n=== Results Summary ===")
    for fold_name, row_count in results.items():
        print(f"{fold_name}: {row_count} rows")
    
    print(f"\nFiltered files have been saved to: {OUTPUT_DIR}")
        


if __name__ == "__main__":
    main()