import torch


def elemtary_to_michaelis_menten_basic(rates: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of elementary rate constants to Michaelis-Menten parameters using vectorized PyTorch operations for GPU compatibility.
    """
    # Unpack for clarity
    k_plus_1, k_minus_1, k_plus_2 = rates.unbind(dim=-1)

    epsilon = 1e-12 # avoid division by 0

    # Calculate kcat, KM, and Ki
    kcat = k_plus_2
    KM = (k_minus_1 + k_plus_2) / (k_plus_1 + epsilon)
    Ki = k_minus_1 / (k_plus_1 + epsilon)

    return torch.stack([kcat, KM, Ki], dim=1)


def elemtary_to_michaelis_menten_advanced(rates: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of elementary rate constants to Michaelis-Menten parameters using vectorized PyTorch operations for GPU compatibility.
    """
    # Unpack for clarity
    k_plus_1, k_minus_1, k_plus_2, k_minus_2, k_plus_3 = rates.unbind(dim=-1)

    epsilon = 1e-12 # avoid division by 0
    common_denominator_part = k_plus_2 + k_minus_2 + k_plus_3

    # Calculate kcat, KM, and Ki
    kcat = (k_plus_2 * k_plus_3) / (common_denominator_part + epsilon)
    km_numerator = (k_minus_1 * k_minus_2) + (k_minus_1 * k_plus_3) + (k_plus_2 * k_plus_3)
    km_denominator = k_plus_1 * common_denominator_part
    KM = km_numerator / (km_denominator + epsilon)
    Ki = k_minus_1 / (k_plus_1 + epsilon)

    return torch.stack([kcat, KM, Ki], dim=1)