# demo_dataloader.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from astra.data_processing.manifests.manifest import create_manifests
from astra.data_processing.datasets import ProteinLigandDataset


def setup_dummy_data():
    """Creates dummy raw data files for demonstration purposes."""
    print("--- Step 1: Setting up dummy data files ---")
    
    # Training data with two valid pairs
    train_data = {
        "protein_sequence": ["MDSKGSSQKGSRLLLLLVVSNLLLCQGVVS", "MAGAASPCPLRLSLPLLLLLLLLLLGPGSG", "PEPTIDE"],
        "ligand_smiles": ["CCO", "CC(=O)O", "CNC"],
        "kcat": [1.0, 3.0, 5.0], "KM": [0.1, 0.3, 0.5], "Ki": [10.0, 30.0, 50.0]
    }
    pd.DataFrame(train_data).to_csv("train_raw.csv", index=False)

    # Validation data with one repeated pair, one new valid pair, and one invalid protein
    val_data = {
        "protein_sequence": ["MDSKGSSQKGSRLLLLLVVSNLLLCQGVVS", "ANOTHERPEPTIDE", "INVALIDPROTEINXXX"],
        "ligand_smiles": ["CCO", "CN", "CCC"],
        "kcat": [1.0, 4.0, 6.0], "KM": [0.1, 0.4, 0.6], "Ki": [10.0, 40.0, 60.0]
    }
    pd.DataFrame(val_data).to_csv("val_raw.csv", index=False)
    print("Dummy files 'train_raw.csv' and 'val_raw.csv' created.")


def main():
    """Main function to run the full demonstration."""
    
    # Create dummy csvs for demo
    setup_dummy_data()

    # IMPORTANT: These keys must be "train", "valid", and/or "test" to use AstraDataModule
    predefined_splits = {"train": "train_raw.csv", "valid": "val_raw.csv"}

    # Output directory for storing and verifying embeddings 
    # TODO: Discuss as a group which dir to store on Balam to avoid having multiple embedding copies
    output_dir = Path("./data")

    # Create dict (i.e. manifest_files) with paths to precomputed protein and ligand embeddings, and target values
    manifest_files = create_manifests(split_files=predefined_splits, output_dir=output_dir)

    # Create dataset from manifest CSV path
    train_path = manifest_files['train']
    train_dataset = ProteinLigandDataset(train_path)
    print(f"Successfully created a dataset with {len(train_dataset)} samples.")
    
    # Create dataloader from dataset
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    first_batch = next(iter(train_dataloader)) # This is a dict

    # Seperate batches by key
    x_prot = first_batch['protein_embedding']
    x_lig = first_batch['ligand_embedding']
    y = first_batch['targets']
    
    # y_hat = model(x_prot, x_lig)
    # loss = loss_func(y, y_hat)


if __name__ == '__main__':
    main()