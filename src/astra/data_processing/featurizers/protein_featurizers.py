import torch
from typing import List, Dict
from transformers import AutoTokenizer, EsmModel

from astra.data_processing.featurizers.base import Featurizer

class ESMFeaturizer(Featurizer):
    """Featurizer for protein sequences using ESM models."""
    def __init__(self, model_name: str, device: torch.device):
        print(f"Loading ESM model: {model_name}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def featurize(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        if not sequences:
            return {}
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1022).to(self.device)
        embeddings = self.model(**inputs).last_hidden_state
        
        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return {seq: emb.cpu() for seq, emb in zip(sequences, mean_embeddings)}