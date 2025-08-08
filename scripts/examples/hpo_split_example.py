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

from astra.constants import EMB_PATH
from astra.data_processing.splits.mmseqs2_split_w_ratio import create_hpo_split_from_fold_balanced


# --- Example Usage ---
if __name__ == '__main__':

    print("Running HPO split creation process...")
    
    # --- Configuration for HPO Split ---
    # 1. Define the input file: the training set from one of your original CV folds.
    fold_train_file = EMB_PATH/"valid_core_filtered/fold_0_train.csv"
    
    # 2. Define the output directory for the new, smaller HPO split files.
    hpo_output_directory = EMB_PATH/"hpo_splits_balanced"
    
    # 3. Define the kinetic parameters. These must be consistent with your project.
    kinetic_params = ['kcat', 'KM', 'Ki']
    
    # 4. Call the new HPO splitting function with the configured parameters.
    create_hpo_split_from_fold_balanced(
        input_train_fold_path=fold_train_file,
        output_dir=hpo_output_directory,
        split_ratio=0.8, # This ratio determines the n_folds used internally.
        seed=42,
        kinetic_cols=kinetic_params,
        max_iterations=1000 # This controls the greedy search; can be lowered for speed.
    )
