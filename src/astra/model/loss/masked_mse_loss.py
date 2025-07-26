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
        # Create a mask for non-NaN values
        mask = ~torch.isnan(targets)
        
        # Move weights to the same device as targets
        self.weights = self.weights.to(targets.device)

        # Calculate MSE only for non-NaN values.
        loss_per_element = F.mse_loss(predictions, targets, reduction='none')
        masked_loss_per_element = loss_per_element * mask
        
        # Sum the loss for each task
        total_loss_per_task = torch.sum(masked_loss_per_element, dim=0)
        
        # Count the number of non-NaN values for each task
        num_valid_samples_per_task = torch.sum(mask, dim=0).float() + 1e-12
        
        # Calculate the mean loss per task
        mean_loss_per_task = total_loss_per_task / num_valid_samples_per_task
        weighted_loss_per_task = mean_loss_per_task * self.weights
        total_loss = torch.sum(weighted_loss_per_task)
        
        return total_loss