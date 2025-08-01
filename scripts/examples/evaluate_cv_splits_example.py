"""
CV Splits Evaluation Example

This script demonstrates how to use the evaluate_splits.py functionality
to check for sequence similarity leakage in k-fold cross-validation splits.

The script evaluates CV splits created by MMseqs2 clustering to ensure:
1. No cluster_id leakage within folds (train vs validation)
2. No cluster_id leakage across different folds
3. Proper distribution of clusters across the dataset

Outputs:
- Comprehensive evaluation report to console
- Optional report file with detailed analysis
- Pass/fail validation result

Usage:
    1. create cv folds with mmseqs2_split_w_ratio_example.py
    2. set TYPE = "core" or "pangenomic"
    3. set CV_FOLDS_DIR = "cv_folds" or "cv_folds_balanced"
    4. run with uv:
        uv run scripts/examples/evaluate_cv_splits_example.py
"""

import sys
import os
from pathlib import Path

# Setup paths
TYPE = "core"  # "pangenomic" or "core"
ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]
ASTRA_DIR = ROOT_DIR / "src" / "astra"
RESULTS_DIR = ROOT_DIR / "results" / f"cv_splits_{TYPE}"
CV_FOLDS_DIR = RESULTS_DIR / "cv_folds_balanced" # "cv_folds" or "cv_folds_balanced"


sys.path.append(str(ASTRA_DIR))

from astra.data_processing.splits.evaluate_splits import evaluate_cv_splits


def run_cv_splits_evaluation():
    """
    Run the CV splits evaluation pipeline.
    
    Returns:
        tuple: (validation_passed: bool, report: str)
    """
    print(f"Evaluating CV splits in: {CV_FOLDS_DIR}")

    
    validation_passed, report = evaluate_cv_splits(
        data_dir=CV_FOLDS_DIR,
        output_file=str(RESULTS_DIR / "cv_splits_evaluation_report.txt"),
        plot_dir=str(CV_FOLDS_DIR / "cv_splits_evaluation_plots"),
        generate_plots=True
    )
    
    return validation_passed, report
    

if __name__ == '__main__':
    validation_passed, report = run_cv_splits_evaluation()