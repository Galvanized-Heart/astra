import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from astra.data_processing.tokenize.tokenizers import LigandTokenizer, ProteinTokenizer

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

def preprocess_and_validate_data(df):
    """Loads data, validates sequences and SMILES, and returns a clean DataFrame."""    
    initial_count = len(df)
    print(f"Loaded {initial_count} rows.")

    # Validate protein sequences
    df['protein_valid'] = df['protein_sequence'].apply(is_valid_protein_sequence)
    
    # Validate SMILES strings
    df['smiles_valid'] = df['ligand_smiles'].apply(is_valid_smiles)

    # Filter out invalid rows
    valid_df = df[df['protein_valid'] & df['smiles_valid']].copy()
    final_count = len(valid_df)
    
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Dropped {dropped_count} invalid rows.")

    # Clean up validation columns
    valid_df = valid_df.drop(columns=['protein_valid', 'smiles_valid'])
    
    return valid_df.reset_index(drop=True)


class ProteinLigandDataset(Dataset):
    """Dataset class for managing protein sequences, ligand SMILES, and target values for predictions."""
    def __init__(self, data_path, protein_tokenizer=None, ligand_tokenizer=None):
        """
        Args:
            data_path (str): Path to the CSV file.
            protein_tokenizer (ProteinTokenizer, optional): A pre-fitted protein tokenizer.
            ligand_tokenizer (LigandTokenizer, optional): A pre-fitted ligand tokenizer.
        """
        df = pd.read_csv(data_path, usecols=["protein_sequence", "ligand_smiles", "kcat", "KM", "Ki"])

        # Validate input data
        valid_df = preprocess_and_validate_data(df)

        protein_sequences = valid_df["protein_sequence"].tolist()
        ligand_smiles = valid_df["ligand_smiles"].tolist()
        self.target_values = torch.tensor(valid_df[["kcat", "KM", "Ki"]].values, dtype=torch.float32)

        # Set protein tokeinzer
        if protein_tokenizer is None:
            print("No protein tokenizer provided. Setting default protein tokenzier.")
            self.protein_tokenizer = ProteinTokenizer()
            self.protein_tokenizer.fit_on_sequences(protein_sequences)
        else:
            self.protein_tokenizer = protein_tokenizer

        # Set ligand tokeinzer
        if ligand_tokenizer is None:
            print("No ligand tokenizer provided. Setting default ligand tokenzier.")
            self.ligand_tokenizer = LigandTokenizer()
            self.ligand_tokenizer.fit_on_sequences(ligand_smiles)
        else:
            self.ligand_tokenizer = ligand_tokenizer

        # Tokenize data
        print("Tokenizing all data...")
        self.protein_encodings = self.protein_tokenizer.batch_encode_plus(protein_sequences)
        self.ligand_encodings = self.ligand_tokenizer.batch_encode_plus(ligand_smiles)
        print("Tokenization complete.")

        # TODO: Implement data featurization

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.target_values)

    def __getitem__(self, idx):
        """
        Returns a single data point from the dataset at the given index.
        The data point is a dictionary of tensors.
        """
        item = {
            "protein_input_ids": self.protein_encodings["input_ids"][idx],
            "protein_attention_mask": self.protein_encodings["attention_mask"][idx],
            "ligand_input_ids": self.ligand_encodings["input_ids"][idx],
            "ligand_attention_mask": self.ligand_encodings["attention_mask"][idx],
            "targets": self.target_values[idx]
        }
        return item