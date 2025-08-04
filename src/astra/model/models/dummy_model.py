import torch
import torch.nn as nn

class DummyModel(nn.Module):
    """
    A dummy model that ignores inputs and returns a zero tensor of shape
    (batch_size, 3). Useful for testing the training pipeline.
    """
    def __init__(self, protein_emb_dim: int = 320, ligand_emb_dim: int = 2048, out_dim: int = 3):
        super().__init__()
        print(protein_emb_dim)
        print(ligand_emb_dim)
        total_input_dim = protein_emb_dim['embedding'][1] + ligand_emb_dim['embedding'][0]
        self.fc1 = nn.Linear(total_input_dim, out_dim)

    def forward(self, protein_embedding: torch.Tensor = None, ligand_embedding: torch.Tensor = None, **kwargs) -> torch.Tensor:
        averaged_protein_embedding = torch.mean(protein_embedding, dim=1)
        print(protein_embedding.shape)
        print(averaged_protein_embedding.shape)
        print(ligand_embedding.shape)

        combined_features = torch.cat((averaged_protein_embedding, ligand_embedding), dim=1)
        return self.fc1(combined_features)