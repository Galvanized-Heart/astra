"""import torch
import torch.nn as nn
from typing import Dict, Any

# This maps the architecture name to the exact layer string name we want to tap
HOOK_REGISTRY = {
    "LinearBaselineModel": "linear.2",         
    "CpiPredConvModel": "fc_2",                
    "CpiPredSelfAttnModel": "encoder_layer.pos_ffn.fc.0", 
    "CpiPredCrossAttnModel": "mlp.0"           
}

class FeatureExtractor:
    """Attaches a forward hook to grab internal representations."""
    def __init__(self, model: nn.Module, layer_name: str):
        self.features = None
        
        # We need to find the specific submodule by its string path (e.g. 'linear.2')
        target_layer = dict(model.named_modules()).get(layer_name)
        if target_layer is None:
            raise ValueError(f"Hook layer '{layer_name}' not found in the model.")
            
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # We detach and move to CPU immediately to save GPU memory
        self.features = output.detach().cpu().numpy()

    def remove(self):
        self.hook_handle.remove()


class InferenceEngine:
    def __init__(self, model: nn.Module, arch_name: str, recomposer: Any):
        self.model = model
        self.recomposer = recomposer
        
        # Attach the hook
        hook_layer_name = HOOK_REGISTRY.get(arch_name)
        if not hook_layer_name:
            raise ValueError(f"Architecture '{arch_name}' not in HOOK_REGISTRY.")
            
        self.extractor = FeatureExtractor(self.model, hook_layer_name)

    def run_batch(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Runs a single batch and returns all standardized data."""
        # 1. Grab raw inputs (averaging protein seq_len dimension to get flat vector)
        raw_prot = batch['protein_embedding'].mean(dim=1).cpu().numpy()
        raw_lig = batch['ligand_embedding'].cpu().numpy()
        
        # 2. Extract targets
        targets = batch.pop("targets")

        # 3. Forward Pass
        raw_output = self.model(
            protein_embedding=batch['protein_embedding'], 
            ligand_embedding=batch['ligand_embedding'],
            protein_attention_mask=batch.get('protein_attention_mask')
        )
        
        # 4. Standardize output via Recomposer
        standardized_outputs = self.recomposer.process(raw_output, targets)
        
        # 5. Grab the hooked features
        learned_features = self.extractor.features

        return {
            "true_targets": standardized_outputs["targets"].cpu().numpy(),
            "predictions": standardized_outputs["preds"].cpu().numpy(),
            "rates": standardized_outputs["rates"].cpu().numpy(),
            "raw_protein_emb": raw_prot,
            "raw_ligand_emb": raw_lig,
            "learned_emb": learned_features
        }

    def cleanup(self):
        """Removes the PyTorch hook to prevent memory leaks."""
        self.extractor.remove()"""

import torch
import torch.nn as nn
from typing import Dict, Any

class InferenceEngine:
    def __init__(self, model: nn.Module, recomposer: Any):
        self.model = model
        self.recomposer = recomposer

    def run_batch(self, batch: dict) -> Dict[str, torch.Tensor]:
        # Grab raw inputs (averaging protein seq_len dimension to get flat vector)
        raw_prot = batch['protein_embedding'].mean(dim=1).cpu().numpy()
        raw_lig = batch['ligand_embedding'].cpu().numpy()
        
        targets = batch.pop("targets")

        # Forward Pass
        raw_output = self.model(
            protein_embedding=batch['protein_embedding'], 
            ligand_embedding=batch['ligand_embedding'],
            protein_attention_mask=batch.get('protein_attention_mask')
        )
        
        # Standardize output
        standardized_outputs = self.recomposer.process(raw_output, targets)

        return {
            "true_targets": standardized_outputs["targets"].cpu().numpy(),
            "predictions": standardized_outputs["preds"].cpu().numpy(),
            "rates": standardized_outputs["rates"].cpu().numpy(),
            "raw_protein_emb": raw_prot,
            "raw_ligand_emb": raw_lig
        }