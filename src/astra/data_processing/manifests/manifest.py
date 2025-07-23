import pandas as pd
import torch
import hashlib
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from safetensors.torch import save_file

from astra.data_processing.featurizers import Featurizer, ESMFeaturizer, MorganFeaturizer
from astra.data_processing.utils import preprocess_and_validate_data


def generate_and_save_features(
    items: List[str],
    featurizer: Featurizer,
    output_dir: Path,
    feature_name: str = "embedding"
) -> Dict[str, str]:
    """
    Generic function to generate, save, and cache features for a list of items.
    """
    item_to_path_map = {}
    items_to_process = []
    
    # Check if embeddings have already been generated item
    print(f"Checking for cached {feature_name} features in {output_dir}...")
    for item in tqdm(items, desc="Checking cache"):
        item_hash = hashlib.sha256(item.encode()).hexdigest()
        output_file = output_dir / f"{item_hash}.safetensors"
        item_to_path_map[item] = str(output_file)
        if not output_file.exists():
            items_to_process.append(item)

    print(f"Found {len(items_to_process)} new items to featurize.")
    if not items_to_process:
        return item_to_path_map

    # Generate features for unsaved items
    print("Generating new features...")
    newly_computed_features = featurizer.featurize(items_to_process)
    
    # Save featurized items
    print("Saving new features to disk...")
    for item, tensor in tqdm(newly_computed_features.items(), desc="Saving"):
        path_str = item_to_path_map[item]
        save_file({feature_name: tensor}, path_str)
        
    return item_to_path_map


def create_manifests(
        split_files: Dict[str, str],
        output_dir: str,
        protein_featurizer: Featurizer,
        ligand_featurizer: Featurizer
    ) -> Dict[str, Path]:
    """
    Generates precomputed features and separate manifests for predefined data splits.

    Args:
        split_files (Dict[str, str]): A dictionary mapping split names (e.g. 'train', 'val')
                                        to their corresponding raw CSV file paths.
        output_dir (str): The directory where manifests and feature sub-folders will be saved.
        protein_featurizer (Featurizer): Protein featurizer object.
        ligand_featurizer (Featurizer): Ligand featurizer object.

    Returns:
        manifest_files (Dict[str, Path]): A dictionary mapping split names (e.g. 'train', 'val')
                                        to their corresponding manifest CSV file paths.
    """
    output_path = Path(output_dir)
    protein_features_dir = output_path / protein_featurizer.name
    ligand_features_dir = output_path / ligand_featurizer.name
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
            print(f"ERROR processing {file_path}: {e}. Aborting.")
            return
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Consolidated {len(full_df)} total rows from {len(split_files)} files.")

    # Filter out invalid and duplicate entries
    clean_df  = preprocess_and_validate_data(full_df, "protein_sequence", "ligand_smiles")

    # Get unique proteins and ligands from colsolidated DataFrame
    unique_proteins = clean_df ["protein_sequence"].dropna().unique()
    unique_ligands = clean_df ["ligand_smiles"].dropna().unique()
    print(f"Found {len(unique_proteins)} unique proteins and {len(unique_ligands)} unique ligands.")

    # Run feature generation process
    print("\n--- Step 3: Generating features ---")
    protein_map = generate_and_save_features(unique_proteins, protein_featurizer, protein_features_dir, "embedding")
    ligand_map = generate_and_save_features(unique_ligands, ligand_featurizer, ligand_features_dir, "embedding")

    # Create final manifests
    print("\n--- Step 4: Creating final manifest files ---")
    manifest_files = {}
    for split_name, file_path in split_files.items():
        print(f"Creating manifest for '{split_name}'...")
        split_df = pd.read_csv(file_path) # Read original file to preserve its structure
        
        # Map sequences and SMILES to their precomputed feature paths
        split_df['protein_feature_path'] = split_df['protein_sequence'].map(protein_map)
        split_df['ligand_feature_path'] = split_df['ligand_smiles'].map(ligand_map)

        # Select final columns and drop any invalid rows
        manifest_df = split_df[['protein_feature_path', 'ligand_feature_path', 'kcat', 'KM', 'Ki']].dropna()

        # Save manifest files and their paths
        manifest_path = output_path / f"manifest_{split_name}.csv"
        manifest_df.to_csv(manifest_path, index=False)
        manifest_files[split_name] = manifest_path
        print(f"Saved {split_name} manifest ({len(manifest_df)} samples) to {manifest_path}")

    print("\nAll manifests created successfully.")

    return manifest_files