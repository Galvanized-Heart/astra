import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('.')
CPI_CSV_PATH = Path.joinpath(PROJECT_ROOT, "data", "interim", "cpipred")
CPI_CSV_PATH.mkdir(parents=True, exist_ok=True) # Ensure directory exists

RENAME_MAP = {
    'CMPD_SMILES': 'ligand_smiles',
    'SEQ': 'protein_sequence',
    'Km': 'KM',
    'kcat_KM': 'kcat/KM'
}

# Core Processing Functions

def load_data(paths):
    """Loads a dictionary of dataframes from a dictionary of paths."""
    dfs = {}
    for name, path in paths.items():
        if path.exists():
            dfs[name] = pd.read_csv(path, index_col=0).rename(columns=RENAME_MAP)
        else:
            print(f"Warning: File not found at {path}. Skipping.")
    return dfs

def merge_kinetic_data(dfs):
    """
    Correctly merges the kinetic dataframes, using different keys for Ki.
    """
    # Define the dataframes to merge, ensuring they exist
    df_kcat = dfs.get('kcat')
    df_km = dfs.get('km')
    df_kcat_km = dfs.get('kcat_km')
    df_ki = dfs.get('ki')
    
    # Use a base dataframe to start the merge. Let's start with kcat or an empty one.
    if df_kcat is None:
        return pd.DataFrame() # Return empty if essential data is missing
        
    merged_df = df_kcat
    
    # Merge with KM and kcat/KM on keys that include 'organism'
    merge_keys_full = ['protein_sequence', 'ligand_smiles', 'organism']
    if df_km is not None:
        merged_df = pd.merge(merged_df, df_km, on=merge_keys_full, how='outer')
    if df_kcat_km is not None:
        merged_df = pd.merge(merged_df, df_kcat_km, on=merge_keys_full, how='outer')
        
    # Merge with Ki on keys that DO NOT include 'organism'
    merge_keys_no_org = ['protein_sequence', 'ligand_smiles']
    if df_ki is not None:
        merged_df = pd.merge(merged_df, df_ki, on=merge_keys_no_org, how='outer')
        
    return merged_df

def verify_counts(merged_df, original_dfs):
    """Verifies that no data points were lost during the merge."""
    print("\n--- Data Integrity Validation ---")
    all_checks_passed = True
    params_to_check = {'kcat': 'kcat', 'km': 'KM', 'kcat_km': 'kcat/KM', 'ki': 'Ki'}
    
    for name, param_col in params_to_check.items():
        if name in original_dfs:
            original_count = original_dfs[name][param_col].notna().sum()
            final_count = merged_df[param_col].notna().sum()
            
            print(f"Validating '{param_col}' counts...")
            print(f"  - Original file rows: {original_count}")
            print(f"  - Merged dataframe rows: {final_count}")

            if original_count == final_count:
                print("MATCH!")
            else:
                print(f"MISMATCH! Data may have been duplicated or lost.")
                all_checks_passed = False
    
    if all_checks_passed:
        print("\nAll data counts successfully validated.")
    else:
        print("\nWARNING: One or more data counts do not match.")

def verify_kcat_km_consistency(df):
    """Checks for logical inconsistencies between kcat, KM, and kcat/KM."""
    print("\n--- Specific Data Anomaly Checks ---")
    # Corrected to use .sum() to get the count
    count1 = (df['kcat/KM'].notna() & (df['kcat'].isna() != df['KM'].isna())).sum()
    print(f"Found {count1} rows with 'kcat/KM' but missing exactly one of 'kcat' or 'KM'.")

    count2 = (df['kcat'].notna() & df['KM'].notna() & df['kcat/KM'].isna()).sum()
    print(f"Found {count2} rows with 'kcat' and 'KM' but missing 'kcat/KM'.")

def enrich_data(df):
    """
    Calculates missing kinetic parameters from available data.
    This function modifies the DataFrame in-place.
    """
    print("\n--- Data Enrichment ---")
    # Initialize provenance columns
    df['kcat_is_calculated'] = False
    df['KM_is_calculated'] = False
    df['kcat/KM_is_calculated'] = False

    # Enrich kcat/KM where kcat and KM are present
    mask1 = (df['kcat'].notna() & df['KM'].notna() & df['kcat/KM'].isna())
    if mask1.sum() > 0:
        print(f"Calculating 'kcat/KM' for {mask1.sum()} rows...")
        df.loc[mask1, 'kcat/KM'] = df.loc[mask1, 'kcat'] / df.loc[mask1, 'KM']
        df.loc[mask1, 'kcat/KM_is_calculated'] = True

    # Enrich kcat where KM and kcat/KM are present
    mask2 = (df['KM'].notna() & df['kcat/KM'].notna() & df['kcat'].isna())
    if mask2.sum() > 0:
        print(f"Calculating 'kcat' for {mask2.sum()} rows...")
        df.loc[mask2, 'kcat'] = df.loc[mask2, 'KM'] * df.loc[mask2, 'kcat/KM']
        df.loc[mask2, 'kcat_is_calculated'] = True

    # Enrich KM where kcat and kcat/KM are present
    mask3 = (df['kcat'].notna() & df['kcat/KM'].notna() & df['KM'].isna() & (df['kcat/KM'] != 0))
    if mask3.sum() > 0:
        print(f"Calculating 'KM' for {mask3.sum()} rows...")
        df.loc[mask3, 'KM'] = df.loc[mask3, 'kcat'] / df.loc[mask3, 'kcat/KM']
        df.loc[mask3, 'KM_is_calculated'] = True
    
    return df

def finalize_and_save(df, output_path):
    """Cleans up the final dataframe and saves it to a CSV."""
    print("\n--- Finalizing and Saving ---")
    final_columns = [
        'protein_sequence', 'ligand_smiles', 'organism',
        'kcat', 'kcat_is_calculated',
        'KM', 'KM_is_calculated',
        'kcat/KM', 'kcat/KM_is_calculated',
        'Ki'
    ]
    # Ensure all columns exist before reindexing
    for col in final_columns:
        if col not in df.columns:
            df[col] = np.nan # Add missing column with NaNs
            
    final_df = df.reindex(columns=final_columns)
    final_df = final_df.drop_duplicates().reset_index(drop=True)

    # Preview and save
    pd.set_option('display.max_columns', None)
    print("\n--- Combined Data Preview ---")
    print(final_df.head(10))
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccess! The fully enriched data has been saved to '{output_path}'")

def process_dataset(name_suffix, output_filename):
    """Main processing pipeline for a given set of files."""
    print(f"\n{'='*20} PROCESSING DATASET: {name_suffix} {'='*20}")
    
    paths = {
        'kcat': CPI_CSV_PATH / f"CPI_kcat_{name_suffix}.csv",
        'km': CPI_CSV_PATH / f"CPI_Km_{name_suffix}.csv",
        'kcat_km': CPI_CSV_PATH / f"CPI_kcat_KM_{name_suffix}.csv",
        'ki': CPI_CSV_PATH / f"CPI_Ki_{name_suffix}.csv"
    }
    
    original_dfs = load_data(paths)
    if not original_dfs:
        print(f"No data found for suffix '{name_suffix}'. Aborting.")
        return

    merged_df = merge_kinetic_data(original_dfs)
    verify_counts(merged_df, original_dfs)
    verify_kcat_km_consistency(merged_df)
    
    enriched_df = enrich_data(merged_df)
    
    output_path = CPI_CSV_PATH / output_filename
    finalize_and_save(enriched_df, output_path)

if __name__ == "__main__":
    process_dataset(name_suffix="core", output_filename="CPI_all_brenda_core_enriched.csv")
    process_dataset(name_suffix="scrn", output_filename="CPI_all_brenda_pangenomic_enriched.csv")