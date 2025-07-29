"""
MMseqs2 Dataset Split Example

Splits sequence datasets into train/validation/test sets while 
avoiding similarity bias with MMseqs2 clustering.

Features:

- Main Split
  - Cluster sequences by identity
  - 80% train, 10% validation, 10% test
  - Keep similar sequences in the same split

- Cross‑Validation
  - Generate 5 folds from the training set
  - Keep clusters intact across folds

- Analysis & Visualization
  - Summarize kinetic parameters (kcat, KM, Ki)
  - Plot cluster size distributions
  - Draw Venn diagrams of parameter overlap

Outputs:
- `train.csv`, `valid.csv`, `test.csv`
- `cv_folds/` (5 cross‑validation splits)
- `results/cv_splits_pangenomic/` (plots and reports)

Requirements:
- MMseqs2 installed and in your PATH  
- Input CSV with sequence and kinetic parameter columns
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]
ASTRA_DIR = ROOT_DIR / "src" / "astra"
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "results" / "cv_splits_pangenomic"

# Add module path
sys.path.append(str(ASTRA_DIR))

from astra.data_processing.splits.mmseqs2_split import mmseqs2_split_data_into_files

def run_mmseqs2_splits():
    # Input data configuration
    input_data = {
        'csv_path': DATA_DIR / "CPI_all_brenda_pangenomic_enriched.csv",
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

    # Split configuration
    split_config = {
        'train': 0.8,
        'valid': 0.1,
        'test': 0.1
    }

    # Cross-validation settings
    cv_config = {
        'n_folds': 5
    }

    # Run MMseqs2 splitting pipeline
    results = mmseqs2_split_data_into_files(
        input_csv_path=input_data['csv_path'],
        output_dir=OUTPUT_DIR,
        seq_col=input_data['sequence_column'],
        seq_id=clustering_params['seq_identity'],
        split_ratios=split_config,
        cluster_size_cap=clustering_params['cluster_size_cap'],
        kinetic_cols=input_data['kinetic_columns'],
        threads=clustering_params['threads'],
        seed=clustering_params['seed'],
        n_folds=cv_config['n_folds']
    )

    return results

if __name__ == '__main__':
    run_mmseqs2_splits()

