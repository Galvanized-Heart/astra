import torch
import torch.nn as nn
from astra.model.modules.attention_utils import *

# Orignial class name from CPI-Pred's Z05X_All_Models.py: SQembSAtt_CPenc_Model
# TODO: Include github link once published

class CpiPredSelfAttnModel(nn.Module):
    """
    An adaptation of the CPI-Pred Self-Attention model (SQembSAtt_CPenc_Model) for the Astra pipeline.

    This model uses multi-head self-attention to process the protein sequence embedding,
    pools the result into a fixed-size vector, and combines it with a ligand fingerprint
    in a final feed-forward network.
    """
    def __init__(self,
                 # --- Dimensions injected by PipelineBuilder ---
                 protein_emb_dim: dict,
                 ligand_emb_dim: dict,
                 out_dim: int,
                 # --- Hyperparameters from config ---
                 d_k: int,
                 n_heads: int,
                 d_v: int,
                 attn_out_dim: int, # Renamed from original 'out_dim' to avoid conflict
                 d_ff: int):
        super().__init__()

        # --- Derive internal dimensions from pipeline specs ---
        d_model_internal = protein_emb_dim['embedding'][1]
        cmpd_dim_internal = ligand_emb_dim['embedding'][0]

        # The core logic is encapsulated in the EncoderLayer
        self.encoder_layer = EncoderLayer_SAtt(
            d_model=d_model_internal,
            d_k=d_k,
            n_heads=n_heads,
            d_v=d_v,
            attn_out_dim=attn_out_dim,
            cmpd_dim=cmpd_dim_internal,
            d_ff=d_ff,
            final_out_dim=out_dim  # Pass the final prediction dim here
        )

    def get_attn_pad_mask(self, seq_mask):
        """Creates a padding mask for the attention mechanism."""
        batch_size, len_q = seq_mask.size()
        _, len_k = seq_mask.size()
        # eq(0) is PAD token, we want to mask it (True)
        pad_attn_mask = seq_mask.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    def forward(self, protein_embedding: torch.Tensor, ligand_embedding: torch.Tensor, protein_attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for the self-attention model.

        Args:
            protein_embedding (torch.Tensor): Shape [batch_size, seq_len, embed_dim]
            ligand_embedding (torch.Tensor): Shape [batch_size, ligand_dim]
            protein_attention_mask (torch.Tensor): Shape [batch_size, seq_len]
            **kwargs: Must contain 'protein_attention_mask' from the featurizer.
        """
        if protein_attention_mask is None:
            raise ValueError("CpiPredSelfAttnModel requires 'protein_attention_mask' to be provided.")

        # Create the self-attention mask from the padding mask
        self_attn_mask = self.get_attn_pad_mask(protein_attention_mask)  # [batch_size, src_len, src_len]

        # The encoder layer handles the rest of the logic
        output = self.encoder_layer(protein_embedding, self_attn_mask, protein_attention_mask, ligand_embedding)
        return output