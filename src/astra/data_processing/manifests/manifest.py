import pandas as pd
import torch
import hashlib
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from safetensors.torch import save_file

from astra.data_processing.featurizers import Featurizer
from astra.data_processing.utils import preprocess_and_validate_data
from astra.constants import EMB_PATH, DATA_PATH


def generate_and_save_features(
    items: List[str],
    featurizer: Featurizer,
    output_dir: Path,
    batch_size: int = 32
) -> Dict[str, str]:
    """
    Generic function to generate, save, and cache features for a list of items.
    This version robustly handles featurizers that return either a single tensor 
    or a dictionary of tensors for each item.
    """
    item_to_path_map = {}
    items_to_process = []
    
    # Check if embeddings have already been generated item
    print(f"Checking for cached features in {output_dir}...")
    for item in tqdm(items, desc="Checking cache"):
        item_hash = hashlib.sha256(item.encode()).hexdigest()
        output_file = output_dir / f"{item_hash}.safetensors"
        item_to_path_map[item] = str(output_file)
        if not output_file.exists():
            items_to_process.append(item)

    print(f"Found {len(items_to_process)} new items to featurize.")
    if not items_to_process:
        # Return if there are no items to process
        return item_to_path_map

    print(f"Generating new features in batches of {batch_size}...")
    
    # Iterate through the items_to_process list in chunks of batch_size
    for i in tqdm(range(0, len(items_to_process), batch_size), desc="Processing Batches"):
        batch_items = items_to_process[i : i + batch_size]
        newly_computed_features = featurizer.featurize(batch_items)
        
        # Save the features for this batch
        for item, features in newly_computed_features.items():
            path_str = item_to_path_map[item]

            # Ensure tensor becomes a dict for saving
            if not isinstance(features, dict):
                features_to_save = {"embedding": features}
            else:
                features_to_save = features
            save_file(features_to_save, path_str)
            
    return item_to_path_map


def create_manifests(
        split_files: Dict[str, str],
        emb_dir: Path = None,
        output_dir: Path = None,
        protein_featurizer: Featurizer = None,
        ligand_featurizer: Featurizer = None,
        batch_size: int = 32
    ) -> Dict[str, Path]:
    """
    Generates precomputed features and separate manifests for predefined data splits.

    Args:
        split_files (Dict[str, str]): A dictionary mapping split names (e.g. 'train', 'val')
                                        to their corresponding raw CSV file paths.
        output_dir (Path): The directory where manifests and feature sub-folders will be saved.
        protein_featurizer (Featurizer): Protein featurizer object.
        ligand_featurizer (Featurizer): Ligand featurizer object.

    Returns:
        manifest_files (Dict[str, Path]): A dictionary mapping split names (e.g. 'train', 'val')
                                        to their corresponding manifest CSV file paths.
    """
    # Set default embedding directory
    if emb_dir is None:
        emb_dir = EMB_PATH
        assert emb_dir.exists(), "Please define an existing path for emb_dir."
    

    # Set default output directory
    if output_dir is None:
        output_dir = DATA_PATH/"manifest"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)


    protein_features_dir = EMB_PATH / protein_featurizer.name
    ligand_features_dir = EMB_PATH / ligand_featurizer.name
    protein_features_dir.mkdir(parents=True, exist_ok=True)
    ligand_features_dir.mkdir(parents=True, exist_ok=True)

    # Append all splits to a single DataFrame to find unique sequences and SMILES
    print("--- Step 1: Consolidating all data splits for efficient processing ---")
    all_dfs = []
    for split_name, file_path in split_files.items():
        print(f"Loading split '{split_name}' from {file_path}")
        try:
            df = pd.read_csv(file_path, usecols=["protein_sequence", "ligand_smiles", "kcat", "KM", "Ki"])
            all_dfs.append(df)

        except (FileNotFoundError, ValueError) as e:
            raise type(e)(f"ERROR processing {file_path}: {e}. Aborting.") from e
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Consolidated {len(full_df)} total rows from {len(split_files)} files.")

    # Filter out invalid and duplicate entries
    clean_df  = preprocess_and_validate_data(full_df, "protein_sequence", "ligand_smiles")

    # Get unique proteins and ligands from colsolidated DataFrame
    unique_proteins = clean_df ["protein_sequence"].dropna().unique()
    unique_ligands = clean_df ["ligand_smiles"].dropna().unique()
    print(f"Found {len(unique_proteins)} unique proteins and {len(unique_ligands)} unique ligands.")

    # Run feature generation process
    print("\n--- Step 2: Generating features ---")
    protein_map = generate_and_save_features(unique_proteins, protein_featurizer, protein_features_dir, batch_size)
    ligand_map = generate_and_save_features(unique_ligands, ligand_featurizer, ligand_features_dir, batch_size)

    # Create final manifests
    print("\n--- Step 3: Creating final manifest files ---")
    manifest_files = {}
    for split_name, file_path in split_files.items():
        print(f"Creating manifest for '{split_name}'...")
        split_df = pd.read_csv(file_path) # Read original file to preserve its structure
        
        # Map sequences and SMILES to their precomputed feature paths
        split_df['protein_feature_path'] = split_df['protein_sequence'].map(protein_map)
        split_df['ligand_feature_path'] = split_df['ligand_smiles'].map(ligand_map)

        # Select final columns
        manifest_df = split_df[['protein_feature_path', 'ligand_feature_path', 'kcat', 'KM', 'Ki']] 

        # Save manifest files and their paths
        manifest_path = output_dir / f"manifest_{split_name}.csv"
        manifest_df.to_csv(manifest_path, index=False)
        manifest_files[split_name] = manifest_path
        print(f"Saved {split_name} manifest ({len(manifest_df)} samples) to {manifest_path}")

    print("\nAll manifests created successfully.")

    return manifest_files