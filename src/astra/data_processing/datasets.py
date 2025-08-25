import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class ProteinLigandDataset(Dataset):
    """
    Dataset that loads pre-computed protein and ligand embeddings
    stored as safetensors files.
    """
    def __init__(self, manifest_path, target_columns: list, target_transform: str = None):
        """
        Args:
            manifest_path (str): Path to CSV file with precomputed safetensor paths.
        """
        print(f"Loading manifest from {manifest_path}...")
        self.manifest = pd.read_csv(manifest_path)
        self.target_transform = target_transform
        self.target_values = torch.tensor(self.manifest[target_columns].values, dtype=torch.float32)
        print("Dataset ready.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.target_values)

    def __getitem__(self, idx):
        """
        Returns a single data point from the dataset at the given index.
        The data point is a dictionary of tensors.
        """
        row = self.manifest.iloc[idx]
        protein_embedding_path = row["protein_feature_path"]
        ligand_embedding_path = row["ligand_feature_path"]

        # Load from safetensors files
        protein_tensors = load_file(str(protein_embedding_path))
        ligand_tensors = load_file(str(ligand_embedding_path))

        # Extract the tensors using the keys we defined during saving
        protein_embedding = protein_tensors["embedding"]
        ligand_embedding = ligand_tensors["embedding"] 

        # Transform data
        targets = self.target_values[idx]
        if self.target_transform == "log10":
            # Add small epsilon to avoid log(0) = -inf
            targets = torch.log10(targets + 1e-12)

        # Create dictionary for multi-modal data
        item = {
            "protein_embedding": protein_embedding,
            "ligand_embedding": ligand_embedding,
            "targets": targets
        }

        # Add protein attention mask if it exists
        protein_attention_mask = protein_tensors.get("attention_mask")
        if protein_attention_mask is not None:
            item["protein_attention_mask"] = protein_attention_mask

        return item