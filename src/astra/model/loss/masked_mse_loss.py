import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self, weights=[1.0/3, 1.0/3, 1.0/3]):
        super().__init__()
        # Using register_buffer makes sure the tensor moves to the correct device
        # automatically when you call .to(device) or .cuda() on the model.
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, predictions, targets):
        """
        Calculates the weighted, masked Mean Squared Error robustly.
        
        Args:
            predictions (torch.Tensor): The model's output, shape (N, 3)
            targets (torch.Tensor): The ground truth labels, shape (N, 3), may contain NaNs.
        
        Returns:
            torch.Tensor: A single scalar loss value.
        """
        # Create a mask for non-NaN values
        mask = ~torch.isnan(targets)
        
        # If target is NaN, replace with model's pred to give (pred - pred)^2 = 0. 
        # This makes the gradient unaffected by the NaN values.
        targets_safe = torch.where(mask, targets, predictions)
        
        # Calculate MSE. NaNs are now gone from the calculation.
        loss_per_element = (predictions - targets_safe)**2
        
        # Sum the loss for each task
        total_loss_per_task = torch.sum(loss_per_element, dim=0)
        
        # Count the number of valid (non-NaN) samples for each task
        # Add a small epsilon to prevent division by zero if a task has no labels in a batch
        num_valid_samples_per_task = torch.sum(mask, dim=0).float() + 1e-9
        
        # Calculate the mean loss per task
        mean_loss_per_task = total_loss_per_task / num_valid_samples_per_task
        
        # Apply weights and sum to get the final scalar loss
        weighted_loss_per_task = mean_loss_per_task * self.weights
        total_loss = torch.sum(weighted_loss_per_task)
        
        return total_loss