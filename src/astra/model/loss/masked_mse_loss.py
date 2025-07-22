import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0]):
        super().__init__()
        # Ensure weights is a tensor on the correct device later
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        """
        Calculates the weighted, masked Mean Squared Error.
        
        Args:
            predictions (torch.Tensor): The model's output, shape (N, 3)
            targets (torch.Tensor): The ground truth labels, shape (N, 3), may contain NaNs.
        
        Returns:
            torch.Tensor: A single scalar loss value.
        """
        # 1. Create a mask for non-NaN values
        # isnan() -> True for NaN, False for valid number
        # ~ (bitwise NOT) inverts it -> True for valid number, False for NaN
        mask = ~torch.isnan(targets)
        
        # Move weights to the same device as targets
        self.weights = self.weights.to(targets.device)

        # 2. Calculate squared error only for non-NaN values.
        # Where the mask is False (i.e., where target is NaN), the loss will be 0.
        loss_per_element = F.mse_loss(predictions, targets, reduction='none')
        masked_loss_per_element = loss_per_element * mask
        
        # 3. Sum the loss for each task (column-wise)
        # Result is a tensor of shape (3,) with the total loss for each task
        total_loss_per_task = torch.sum(masked_loss_per_element, dim=0)
        
        # 4. Count the number of non-NaN values for each task
        # Add a small epsilon to avoid division by zero if a batch has no labels for a task
        num_valid_samples_per_task = torch.sum(mask, dim=0).float() + 1e-8
        
        # 5. Calculate the mean loss per task
        mean_loss_per_task = total_loss_per_task / num_valid_samples_per_task
        weighted_loss_per_task = mean_loss_per_task * self.weights
        total_loss = torch.sum(weighted_loss_per_task)
        
        return total_loss