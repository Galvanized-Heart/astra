import torch
from astra.model.modules.log10_stable_recomposition import elemtary_to_michaelis_menten_basic_logspace, elemtary_to_michaelis_menten_advanced_logspace

class Recomposer:
    """
    Standardizes model outputs based on the experiment mode.
    """
    def __init__(self, recomp_func_name: str, target_columns: list):
        self.recomp_func_name = recomp_func_name
        self.target_columns = target_columns
        self.is_single_task = len(target_columns) == 1

    def process(self, raw_output: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Takes the raw model output and standardizes it into predictions, rates, and errors.
        """
        batch_size = raw_output.shape[0]
        result = {
            "preds": torch.full((batch_size, 3), float('nan'), device=raw_output.device),
            "rates": torch.full((batch_size, 5), float('nan'), device=raw_output.device),
            "targets": torch.full((batch_size, 3), float('nan'), device=raw_output.device)
        }

        # Map targets to standard 3-column format [kcat, KM, Ki]
        param_to_idx = {'kcat': 0, 'KM': 1, 'Ki': 2}
        for i, param_name in enumerate(self.target_columns):
            idx = param_to_idx[param_name]
            result["targets"][:, idx] = targets[:, i] if self.is_single_task else targets[:, idx]

        # Process Model Output
        if self.recomp_func_name == "AdvancedRecomp":
            result["rates"][:, :5] = raw_output
            result["preds"] = elemtary_to_michaelis_menten_advanced_logspace(raw_output)
            
        elif self.recomp_func_name == "BasicRecomp":
            result["rates"][:, :3] = raw_output
            result["preds"] = elemtary_to_michaelis_menten_basic_logspace(raw_output)
            
        else:
            # Single Task or Direct MT (no recomposition)
            for i, param_name in enumerate(self.target_columns):
                idx = param_to_idx[param_name]
                result["preds"][:, idx] = raw_output[:, i]

        return result