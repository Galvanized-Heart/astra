import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, EsmModel

from astra.data_processing.featurizers.base import Featurizer

class ESMFeaturizer(Featurizer):
    """
    Featurizer for protein sequences using ESM models, providing per-token embeddings.
    """
    def __init__(self, model_name: str, device: torch.device, max_length: int = 1022):
        print(f"Loading ESM model: {model_name}")
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return self.model_name.replace("/", "_") + "_per_token"

    @property
    def feature_spec(self) -> Dict[str, Tuple[int, ...]]:
        """Returns the specification for the per-token embeddings and attention mask."""
        embedding_dim = self.model.config.hidden_size
        return {
            "embedding": (self.max_length, embedding_dim),
            "attention_mask": (self.max_length,)
        }

    @torch.no_grad()
    def featurize(self, sequences: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        if not sequences:
            return {}
        
        inputs = self.tokenizer(sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).to(self.device)
        
        embeddings = self.model(**inputs).last_hidden_state
        attention_mask = inputs['attention_mask']
        
        return {
            seq: {
                "embedding": emb.cpu(),
                "attention_mask": mask.cpu()
            }
            for seq, emb, mask in zip(sequences, embeddings, attention_mask)
        }