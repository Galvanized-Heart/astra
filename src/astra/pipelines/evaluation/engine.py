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