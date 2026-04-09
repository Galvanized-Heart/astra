import torch
import torch.nn as nn

class LinearBaselineModel(nn.Module):
    """
    A dummy model that ignores inputs and returns a zero tensor of shape
    (batch_size, 3). Useful for testing the training pipeline.
    """
    def __init__(self, protein_emb_dim: None, ligand_emb_dim: None, dim_1: int = None, dim_2: int = None, out_dim: int = 3):
        super().__init__()
        total_input_dim = protein_emb_dim['embedding'][1] + ligand_emb_dim['embedding'][0]
        self.linear = nn.Sequential(
            nn.Linear(total_input_dim, dim_1),
            nn.ReLU(),
            nn.Linear(dim_1, dim_2),
            nn.ReLU(),
            nn.Linear(dim_2, out_dim)
        )

    def forward(self, protein_embedding: torch.Tensor = None, ligand_embedding: torch.Tensor = None, **kwargs) -> torch.Tensor:
        averaged_protein_embedding = torch.mean(protein_embedding, dim=1)
        combined_features = torch.cat((averaged_protein_embedding, ligand_embedding), dim=1)
        return self.linear(combined_features)


class ProteinOnlyLinearModel(nn.Module):
    """Ablation model with only protein embeddings."""
    def __init__(self, protein_emb_dim: dict, dim_1: int = 512, dim_2: int = 128, out_dim: int = 3, **kwargs):
        super().__init__()
        # Only use the protein embedding dimension
        input_dim = protein_emb_dim['embedding'][1]
        self.linear = nn.Sequential(
            nn.Linear(input_dim, dim_1),
            nn.ReLU(),
            nn.Linear(dim_1, dim_2),
            nn.ReLU(),
            nn.Linear(dim_2, out_dim)
        )

    def forward(self, protein_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        # Ignore ligand entirely
        averaged_protein_embedding = torch.mean(protein_embedding, dim=1)
        return self.linear(averaged_protein_embedding)


class LigandOnlyLinearModel(nn.Module):
    """Ablation model with only ligand embeddings."""
    def __init__(self, ligand_emb_dim: dict, dim_1: int = 512, dim_2: int = 128, out_dim: int = 3, **kwargs):
        super().__init__()
        # Only use the ligand embedding dimension
        input_dim = ligand_emb_dim['embedding'][0]
        self.linear = nn.Sequential(
            nn.Linear(input_dim, dim_1),
            nn.ReLU(),
            nn.Linear(dim_1, dim_2),
            nn.ReLU(),
            nn.Linear(dim_2, out_dim)
        )

    def forward(self, ligand_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        # Ignore protein entirely
        return self.linear(ligand_embedding)