import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYUX")

def is_valid_protein_sequence(seq):
    """Checks if a protein sequence contains only valid amino acid characters."""
    if not isinstance(seq, str) or not seq:
        return False
    return set(seq.upper()) <= VALID_AMINO_ACIDS

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def preprocess_and_validate_data(df, prot_col, lig_col):
    """Loads data, validates sequences and SMILES, and returns a clean DataFrame."""    
    initial_count = len(df)
    print(f"Loaded {initial_count} rows.")

    # Validate protein sequences
    df['protein_valid'] = df[prot_col].apply(is_valid_protein_sequence)
    
    # Validate SMILES strings
    df['smiles_valid'] = df[lig_col].apply(is_valid_smiles)

    # Filter out invalid rows
    valid_df = df[df['protein_valid'] & df['smiles_valid']].copy()
    final_count = len(valid_df)
    
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Dropped {dropped_count} invalid rows.")

    # Clean up validation columns
    valid_df = valid_df.drop(columns=['protein_valid', 'smiles_valid'])
    
    return valid_df.reset_index(drop=True)


def create_feature_manifest(
        input_csv_path: str, 
        output_dir: str, 
        prot_col: str = "protein_sequence", 
        lig_col: str = "ligand_smiles", 
        kcat_col: str = "kcat", 
        KM_col: str = "KM", 
        Ki_col: str = "Ki"
        ):
    """
    Generates pre-computed features for proteins and ligands, saving them to disk
    and creating a manifest file that maps data points to these features.

    Args:
        input_csv_path (str): Path to the raw input CSV file.
        output_dir (str): Directory where the manifest and feature sub-folders will be saved.
    """
    # Setup feature storage directory
    print("--- Step 1: Loading and validating data ---")
    output_path = Path(output_dir)
    protein_features_dir = output_path / "protein_features"
    ligand_features_dir = output_path / "ligand_features"
    protein_features_dir.mkdir(parents=True, exist_ok=True)
    ligand_features_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    df = pd.read_csv(input_csv_path, usecols=[prot_col, lig_col, kcat_col, KM_col, Ki_col])

    # Filter out invalid and duplicate entries
    df = preprocess_and_validate_data(df, prot_col, lig_col)
    
    # Setup models for featurization
    print("\n--- Step 2: Setting up feature generation models ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ESM2 model for protein embeddings
    print("Loading ESM2 model and tokenizer...")
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
    esm_model.eval() # Set model to evaluation mode

    # Process unique proteins
    print("\n--- Step 3: Generating protein embeddings ---")
    unique_proteins = df["protein_sequence"].unique()
    protein_map = {} # Dictionary to map sequence to its feature file path

    with torch.no_grad():
        for seq in tqdm(unique_proteins, desc="Processing Proteins"):
            # Create a unique, file-system-safe name using a hash
            seq_hash = hashlib.sha256(seq.encode()).hexdigest()
            output_file = protein_features_dir / f"{seq_hash}.safetensors" 

            # Generate features only if they don't already exist
            if not output_file.exists():
                inputs = esm_tokenizer(seq, return_tensors="pt", truncation=True, max_length=1022).to(device)
                outputs = esm_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
                
                # Save as safetensor
                tensors_to_save = {"embedding": embedding}
                save_file(tensors_to_save, output_file)

            protein_map[seq] = str(output_file)

    # Process unique Ligands
    print("\n--- Step 4: Generating ligand fingerprints ---")
    unique_ligands = df["ligand_smiles"].unique()
    ligand_map = {} # Dictionary to map SMILES to its feature file path

    for smiles in tqdm(unique_ligands, desc="Processing Ligands"):
        smiles_hash = hashlib.sha256(smiles.encode()).hexdigest()
        output_file = ligand_features_dir / f"{smiles_hash}.pt"

        if not output_file.exists():
            mol = Chem.MolFromSmiles(smiles)
            # Use a common configuration for Morgan Fingerprints
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_np = np.array(fp, dtype=np.float32)
            fp_tensor = torch.from_numpy(fp_np)
            tensors_to_save = {"fingerprint": fp_tensor}
            save_file(tensors_to_save, output_file)

        ligand_map[smiles] = str(output_file)

    # Create and save the manifest file
    print("\n--- Step 5: Creating and saving the manifest file ---")
    
    # Map the original sequences/SMILES to their new feature file paths
    df['protein_feature_path'] = df['protein_sequence'].map(protein_map)
    df['ligand_feature_path'] = df['ligand_smiles'].map(ligand_map)

    # Select final columns and save
    manifest_df = df[['protein_feature_path', 'ligand_feature_path', 'kcat', 'KM', 'Ki']]
    manifest_path = output_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\nComplete! Manifest saved to: {manifest_path}")
    print(f"Protein features saved in: {protein_features_dir}")
    print(f"Ligand features saved in: {ligand_features_dir}")