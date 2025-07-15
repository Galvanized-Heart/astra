import pandas as pd
import torch
from torch.utils.data import Dataset


class CpiPredDataset(Dataset):
    def __init__(self, csv_file, protein_tokenizer=None, ligand_tokenizer=None):
        # TODO: hardcoded columns should be replaced with constants
        df = pd.read_csv(csv_file, usecols=["protein_sequence", "ligand_smiles", "kcat", "KM", "Ki"])
        protein_sequence = df["protein_sequence"].tolist()
        ligand_smiles = df["ligand_smiles"].values
        self.target_values = torch.tensor(df[["kcat", "KM", "Ki"]].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass