import torch

def log10_sum_exp(log_terms: torch.Tensor) -> torch.Tensor:
    """
    Computes log10(sum(10**x_i)) in a numerically stable way.
    
    Args:
        log_terms (torch.Tensor): A tensor where each element is a log10 term.
                                  Example: torch.stack([log10(a), log10(b), log10(c)], dim=-1)
    
    Returns:
        torch.Tensor: The result of log10(a + b + c).
    """
    # Find the maximum value along the dimension of the terms
    max_log_term, _ = torch.max(log_terms, dim=-1, keepdim=True)
    
    # Subtract the max for stability, compute the sum in linear space, then convert back
    # log10(sum(10^x_i)) = max_log + log10(sum(10^(x_i - max_log)))
    stable_sum = torch.sum(torch.pow(10, log_terms - max_log_term), dim=-1)
    
    # Add the max back in log space
    return max_log_term.squeeze(-1) + torch.log10(stable_sum)



def elemtary_to_michaelis_menten_basic(rates: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of three log10 scaled elementary rate constants to 
    log10 scaled Michaelis-Menten parameters, operating entirely in log-space.
    
    Args:
        rates (torch.Tensor): Tensor of shape [batch, 3] representing 
                                  [log10(k+1), log10(k-1), log10(k+2)].
    
    Returns:
        torch.Tensor: Tensor of shape [batch, 3] containing 
                      [log10(kcat), log10(KM), log10(Ki)].
    """
    # Unpack for clarity
    log_k_plus_1, log_k_minus_1, log_k_plus_2 = rates.unbind(dim=-1)

    # 1. Calculate log10(kcat)
    # kcat = k+2  =>  log10(kcat) = log10(k+2)
    log_kcat = log_k_plus_2

    # 2. Calculate log10(Ki)
    # Ki = k-1 / k+1  =>  log10(Ki) = log10(k-1) - log10(k+1)
    log_Ki = log_k_minus_1 - log_k_plus_1
    
    # 3. Calculate log10(KM)
    # KM = (k-1 + k+2) / k+1  =>  log10(KM) = log10(k-1 + k+2) - log10(k+1)
    # We use log10_sum_exp for the numerator
    log_km_numerator = log10_sum_exp(torch.stack([log_k_minus_1, log_k_plus_2], dim=-1))
    log_KM = log_km_numerator - log_k_plus_1

    output = torch.stack([log_kcat, log_KM, log_Ki], dim=1)

    return output



def elemtary_to_michaelis_menten_advanced(rates: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of five log10 scaled elementary rate constants to 
    log10 scaled Michaelis-Menten parameters, operating entirely in log-space.
    
    Args:
        rates (torch.Tensor): Tensor of shape [batch, 5] representing 
                                  [log10(k+1), log10(k-1), log10(k+2), log10(k-2), log10(k+3)].

    Returns:
        torch.Tensor: Tensor of shape [batch, 3] containing 
                      [log10(kcat), log10(KM), log10(Ki)].
    """
    # Unpack for clarity
    log_k_p1, log_k_m1, log_k_p2, log_k_m2, log_k_p3 = rates.unbind(dim=-1)

    # --- Common Denominator Part: log10(k+2 + k-2 + k+3) ---
    log_common_denom_part = log10_sum_exp(torch.stack([log_k_p2, log_k_m2, log_k_p3], dim=-1))

    # 1. Calculate log10(kcat)
    # kcat = (k+2 * k+3) / (k+2 + k-2 + k+3)
    # log(kcat) = log(k+2) + log(k+3) - log(common_denom)
    log_kcat = (log_k_p2 + log_k_p3) - log_common_denom_part

    # 2. Calculate log10(Ki)
    # Ki = k-1 / k+1
    log_Ki = log_k_m1 - log_k_p1
    
    # 3. Calculate log10(KM)
    # KM = num / den
    # log(KM) = log(num) - log(den)

    # Numerator: log10((k-1*k-2) + (k-1*k+3) + (k+2*k+3))
    term1 = log_k_m1 + log_k_m2
    term2 = log_k_m1 + log_k_p3
    term3 = log_k_p2 + log_k_p3
    log_km_numerator = log10_sum_exp(torch.stack([term1, term2, term3], dim=-1))

    # Denominator: log10(k+1 * (k+2 + k-2 + k+3))
    # log(den) = log(k+1) + log(common_denom)
    log_km_denominator = log_k_p1 + log_common_denom_part
    
    log_KM = log_km_numerator - log_km_denominator
    
    output = torch.stack([log_kcat, log_KM, log_Ki], dim=1)

    return output


"""def elemtary_to_michaelis_menten_basic(rates: torch.Tensor, log_transform: bool = False) -> torch.Tensor:
    "
    Converts a batch of three elementary rate constants to Michaelis-Menten parameters using vectorized PyTorch operations for GPU compatibility.
    "
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
    "
    Converts a batch of five elementary rate constants to Michaelis-Menten parameters using vectorized PyTorch operations for GPU compatibility.
    "

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


"""