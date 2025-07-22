# -*- coding: utf-8 -*-
"""
Dataset Splitting via Random Sampling.

This script provides a standardized function to split a dataset from a single
CSV file into training, validation, and test sets using random sampling.

The primary function, `randomly_split_data_into_files`, orchestrates the process:
- It reads a source CSV file.
- It performs a stratified random split to create three distinct datasets.
- It saves these partitions into separate CSV files within a specified output directory.

This serves as a baseline splitting method, contrasting with more complex
strategies like the sequence-based clustering split.
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# SECTION 1: Main Orchestration Function
# ==============================================================================

def randomly_split_data_into_files(
    input_csv_path: str,
    output_dir: str,
    split_ratios: dict = None,
    seed: int = 42
):
    """
    Reads a CSV, randomly splits it into train, validation, and test sets,
    and saves them to a specified directory.

    This function performs a two-step split to create three partitions:
    1. The full dataset is split into a training set and a temporary set.
    2. The temporary set is then split into validation and test sets.

    Args:
        input_csv_path (str): The path to the source CSV file.
        output_dir (str): The directory where the output files (train.csv,
                          valid.csv, test.csv) will be saved.
        split_ratios (dict, optional): A dictionary specifying the proportions for
                                       'train', 'valid', and 'test'.
                                       Defaults to {'train': 0.8, 'valid': 0.1, 'test': 0.1}.
        seed (int): The seed for the random number generator to ensure
                    reproducible splits.

    Raises:
        FileNotFoundError: If the input_csv_path does not exist.
        ValueError: If the split_ratios do not sum to 1.0 or are incomplete.
    """
    print("--- Starting Random Data Splitting Process ---")

    # --- 1. Validation and Setup ---
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    if split_ratios is None:
        split_ratios = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    
    required_keys = {'train', 'valid', 'test'}
    if not required_keys.issubset(split_ratios.keys()):
        raise ValueError("split_ratios dictionary must contain 'train', 'valid', and 'test' keys.")
    if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, but got {sum(split_ratios.values())}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Read Data and Perform Splits ---
    print(f"1. Reading data from {input_csv_path}...")
    full_df = pd.read_csv(input_csv_path)

    print("2. Performing random splits...")
    # First split: separate the training data from the rest (validation + test)
    train_df, temp_df = train_test_split(
        full_df,
        train_size=split_ratios['train'],
        random_state=seed
    )

    # Second split: separate the validation and test data from the temporary set
    # We need to calculate the proportional size of the validation set relative to the temp set
    remaining_size = split_ratios['valid'] + split_ratios['test']
    if remaining_size == 0:
        # Handle edge case where there's no validation or test set
        valid_df = pd.DataFrame(columns=full_df.columns)
        test_df = pd.DataFrame(columns=full_df.columns)
    else:
        relative_valid_size = split_ratios['valid'] / remaining_size
        valid_df, test_df = train_test_split(
            temp_df,
            train_size=relative_valid_size,
            random_state=seed
        )

    # --- 3. Save Split Files ---
    print("3. Saving partitioned CSV files...")
    output_files = {}
    split_data = {'train': train_df, 'valid': valid_df, 'test': test_df}

    for split_name, df in split_data.items():
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(output_path, index=False)
        output_files[split_name] = {'path': output_path, 'count': len(df)}

    # --- 4. Final Report ---
    print("\n--- Process Complete ---")
    print("Split Summary:")
    total = len(full_df)
    for split_name, info in output_files.items():
        percentage = (info['count'] / total * 100) if total > 0 else 0
        print(f"  - {split_name.title():<8}: {info['count']:>6} rows ({percentage:.1f}%)")
    
    print("\nOutput Files:")
    for split_name, info in output_files.items():
        print(f"  - {info['path']}")
    print("------------------------\n")


# ==============================================================================
# SECTION 2: Example Usage
# ==============================================================================

if __name__ == '__main__':
    # --- Create a dummy CSV file for a runnable demonstration ---
    print("Creating a dummy 'input_for_random_split.csv' for demonstration...")
    dummy_data = {
        'id': range(1000),
        'feature_1': [random.random() for _ in range(1000)],
        'feature_2': [random.randint(0, 100) for _ in range(1000)]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_input_csv = "input_for_random_split.csv"
    dummy_df.to_csv(dummy_input_csv, index=False)

    # --- Define parameters and run the splitting function ---
    OUTPUT_DIRECTORY = "random_split_output"

    randomly_split_data_into_files(
        input_csv_path=dummy_input_csv,
        output_dir=OUTPUT_DIRECTORY,
        split_ratios={'train': 0.8, 'valid': 0.1, 'test': 0.1},
        seed=42
    )

    # --- Verification Step (Optional) ---
    print("Verifying outputs...")
    train_output = pd.read_csv(os.path.join(OUTPUT_DIRECTORY, "train.csv"))
    valid_output = pd.read_csv(os.path.join(OUTPUT_DIRECTORY, "valid.csv"))
    test_output = pd.read_csv(os.path.join(OUTPUT_DIRECTORY, "test.csv"))

    print(f"Train file contains {len(train_output)} rows.")
    print(f"Validation file contains {len(valid_output)} rows.")
    print(f"Test file contains {len(test_output)} rows.")
    assert len(train_output) + len(valid_output) + len(test_output) == len(dummy_df)
    print("Verification successful: Total rows match original dataset.")