# Numerically stable vectorized reimplementation of regression loss from Multi-task learning using uncertainty to weigh losses for 
# scene geometry and semantics by Kendall, Alex, Yarin Gal, and Roberto Cipolla in 2017. https://doi.org/10.48550/arXiv.1705.07115

# Formula:
# total_loss = sum(0.5 * exp(-s) * loss + 0.5 * s)

import torch
import torch.nn as nn

class MaskedUncertaintyMSELoss(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        # Since s=log(sigma^2), s=0 when sigma=1
        # Initialize s=0 for equal initial weighting
        self.log_variance = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, predictions, targets):
        # Ensure correct device
        device = predictions.device

        # Create a mask for non-NaN values
        mask = ~torch.isnan(targets)
        
        # If target is NaN, replace with model's pred to give (pred - pred)^2 = 0.
        # This makes the gradient unaffected by the NaN values
        targets_safe = torch.where(mask, targets, predictions)

        # Calculate MSE. NaNs are now gone from the calculation.
        loss_per_element = (predictions - targets_safe)**2

        # Mask and sum the loss for each task
        total_loss_per_task = torch.sum(loss_per_element * mask.float(), dim=0)

        # Count number of valid samples per task
        num_valid_samples = torch.sum(mask.float(), dim=0)

        # Create mask for tasks with 0 samples in a batch to avoid div by 0
        has_data_mask = (num_valid_samples > 0).float()

        # Avoid division by 0 for mean calculation
        num_valid_samples_per_task = num_valid_samples + (1 - has_data_mask)

        # Calculate the mean loss per task
        mean_loss_per_task = total_loss_per_task / num_valid_samples_per_task

        # Apply parameterized weights to each loss
        stds = self.log_variance.to(device)
        precision = torch.exp(-stds)

        # Calculate loss per task
        task_losses = 0.5 * (precision * mean_loss_per_task + stds)

        # Zero out tasks with no data in a batch
        masked_task_losses = task_losses * has_data_mask

        # Sum total loss together
        total_loss = masked_task_losses.sum()
    
        return total_loss