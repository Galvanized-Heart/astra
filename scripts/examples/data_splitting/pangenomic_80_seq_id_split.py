import sys
from pathlib import Path
from astra.constants import PROJECT_ROOT

from astra.data_processing.splits.mmseqs2_split_w_ratio import mmseqs2_split_data_w_ratio_into_files
from astra.data_processing.splits.filter_valid_core import filter_valid_core_pipeline
from astra.data_processing.splits.mmseqs2_split_w_ratio import create_hpo_split_from_fold_balanced

# Setup paths
TYPE = "pangenomic" # "pangenomic" or "core"
BALANCED_CV = True
SEQ_ID = 1.0

ASTRA_DIR = PROJECT_ROOT / "src" / "astra"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / f"cv_splits_{TYPE}_{SEQ_ID}"

CV_DIR = SPLIT_DIR / "cv_folds_balanced"
FILT_CV_DIR = SPLIT_DIR / "valid_core_filtered"

CREATE_HPO = False
FOLD_USED_FOR_HPO = 0
HPO_DIR = SPLIT_DIR / "hpo"

# Add ASTRA_DIR to Python path for imports
sys.path.append(str(ASTRA_DIR))

def main():
    # Run MMseqs2 splitting pipeline
    print("Running MMseqs2 splitting pipeline with ratio...")
    results = mmseqs2_split_data_w_ratio_into_files(
        input_csv_path=DATA_DIR / f"CPI_all_brenda_{TYPE}_enriched.csv",
        output_dir=SPLIT_DIR,
        seq_col="protein_sequence",
        seq_id=SEQ_ID,
        cluster_size_cap=50,
        kinetic_cols=['kcat', 'KM', 'Ki'],
        threads=4,
        seed=42,
        n_folds=5,
        use_balanced_cv=BALANCED_CV,
        max_cv_iterations=1000,
        compare_cv_methods=True
    )

    # Run validation set filtering pipeline
    print(f"\nRunning core sequence filtering pipeline in: {DATA_DIR}...")
    results = filter_valid_core_pipeline(
        data_dir=DATA_DIR,
        cv_dir=CV_DIR,
        output_dir=FILT_CV_DIR,
        core_filename='CPI_all_brenda_core_enriched.csv',
        pangenomic_filename='CPI_all_brenda_pangenomic_enriched.csv',
        fold_pattern="fold_{}_valid.csv",
        num_folds=5,
        sequence_column='protein_sequence'
    )
    
    # Display results summary
    print("\nResults Summary:\n")
    for fold_name, row_count in results.items():
        print(f"{fold_name}: {row_count} rows")
    
    print(f"\nFiltered files have been saved to: {FILT_CV_DIR}")
    
    if CREATE_HPO:
        # Run HPO splitting if necessary
        create_hpo_split_from_fold_balanced(
            input_train_fold_path=CV_DIR / f"fold_{FOLD_USED_FOR_HPO}_train.csv",
            output_dir=HPO_DIR,
            split_ratio=0.8, # This ratio determines the n_folds used internally.
            seed=42,
            kinetic_cols=['kcat', 'KM', 'Ki'],
            max_iterations=1000
        )

        print(f"\nHPO files have been saved to: {HPO_DIR}")



if __name__ == "__main__":
    main()
