import torch

# TODO: Make sure this is compatible with log10 in DataLoader()

def elemtary_to_michaelis_menten_basic(rates: torch.Tensor, log_transform: bool = False) -> torch.Tensor:
    """
    Converts a batch of three elementary rate constants to Michaelis-Menten parameters using vectorized PyTorch operations for GPU compatibility.
    """
    # 
    if log_transform:
        rates = torch.pow(10, rates)

    # Unpack for clarity
    k_plus_1, k_minus_1, k_plus_2 = rates.unbind(dim=-1)

    # Calculate kcat, KM, and Ki
    kcat = k_plus_2
    KM = (k_minus_1 + k_plus_2) / (k_plus_1 + 1e-12)
    Ki = k_minus_1 / (k_plus_1 + 1e-12)

    mm_params = torch.stack([kcat, KM, Ki], dim=1)
    
    if log_transform:
        mm_params = torch.log10(mm_params + 1e-20)

    return mm_params

def elemtary_to_michaelis_menten_advanced(rates: torch.Tensor, log_transform: bool = False) -> torch.Tensor:
    """
    Converts a batch of five elementary rate constants to Michaelis-Menten parameters using vectorized PyTorch operations for GPU compatibility.
    """

    if log_transform:
        rates = torch.pow(10, rates)

    # Unpack for clarity
    k_plus_1, k_minus_1, k_plus_2, k_minus_2, k_plus_3 = rates.unbind(dim=-1)

    common_denominator_part = k_plus_2 + k_minus_2 + k_plus_3

    # Calculate kcat, KM, and Ki
    kcat = (k_plus_2 * k_plus_3) / (common_denominator_part + 1e-12)
    km_numerator = (k_minus_1 * k_minus_2) + (k_minus_1 * k_plus_3) + (k_plus_2 * k_plus_3)
    km_denominator = k_plus_1 * common_denominator_part
    KM = km_numerator / (km_denominator + 1e-12)
    Ki = k_minus_1 / (k_plus_1 + 1e-12)

    mm_params = torch.stack([kcat, KM, Ki], dim=1)
    
    if log_transform:
        mm_params = torch.log10(mm_params + 1e-20)

    return mm_params


