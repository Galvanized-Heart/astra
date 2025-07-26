import torch
import torch.nn as nn

class DummyModel(nn.Module):
    """
    A dummy model that ignores inputs and returns a zero tensor of shape
    (batch_size, 3). Useful for testing the training pipeline.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_prot: torch.Tensor, x_lig: torch.Tensor) -> torch.Tensor:
        batch_size = x_prot.shape[0]
        return torch.zeros((batch_size, 3), device=x_prot.device)