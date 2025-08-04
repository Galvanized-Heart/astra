"""
MMseqs2 Dataset Split Example with Balanced Cross-Validation

This script demonstrates the enhanced MMseqs2 splitting functionality that includes
balanced cross-validation with kinetic parameter distribution preservation.

The balanced CV approach uses greedy optimization to minimize deviation from the
original dataset's kinetic parameter distribution (kcat, KM, Ki) when creating
cross-validation folds. This ensures that each fold maintains similar proportions
of samples with different kinetic parameters, leading to more reliable model
evaluation.

Outputs:
- Original CV folds (cv_folds/)
- Balanced CV folds (cv_folds_balanced/) 
- Kinetic parameter overlap visualizations (Venn diagrams)
- Distribution quality metrics for comparison

Usage:
    1. set TYPE = "core" or "pangenomic"
    2. set BALANCED_CV = True or False
    2. run with uv:
        uv run scripts/examples/mmseqs2_splits_w_ratio_example.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
TYPE = "pangenomic" # "pangenomic" or "core"
BALANCED_CV = True # True or False
ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]
ASTRA_DIR = ROOT_DIR / "src" / "astra"
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "results" / f"cv_splits_{TYPE}"

sys.path.append(str(ASTRA_DIR))

from astra.data_processing.splits.mmseqs2_split_w_ratio import mmseqs2_split_data_w_ratio_into_files


def run_mmseqs2_splits_w_ratio():
    # Input data configuration
    input_data = {
        'csv_path': DATA_DIR / f"CPI_all_brenda_{TYPE}_enriched.csv",
        'sequence_column': "protein_sequence",
        'kinetic_columns': ['kcat', 'KM', 'Ki']
    }

    # Clustering parameters
    clustering_params = {
        'seq_identity': 0.8,
        'cluster_size_cap': 50,
        'threads': 4,
        'seed': 42
    }
    # TODO: add coverage mode parameter (currently set to 0: bidirectional)
    # TODO: add cluster mode parameter (currently set to 0: greedy set cover)

    # Cross-validation settings
    cv_config = {
        'n_folds': 5
    }

    # Run MMseqs2 splitting pipeline
    print("Running MMseqs2 splitting pipeline with ratio...")
    results = mmseqs2_split_data_w_ratio_into_files(
        input_csv_path=input_data['csv_path'],
        output_dir=OUTPUT_DIR,
        seq_col=input_data['sequence_column'],
        seq_id=clustering_params['seq_identity'],
        cluster_size_cap=clustering_params['cluster_size_cap'],
        kinetic_cols=input_data['kinetic_columns'],
        threads=clustering_params['threads'],
        seed=clustering_params['seed'],
        n_folds=cv_config['n_folds'],
        use_balanced_cv=BALANCED_CV,
        max_cv_iterations=1000,
        compare_cv_methods=True
    )

    return results

if __name__ == '__main__':
    run_mmseqs2_splits_w_ratio()
