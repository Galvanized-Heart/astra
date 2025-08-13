import torch
import torch.nn as nn
from astra.model.modules.attention_utils import *

class CpiPredCrossAttnModel(nn.Module):
    def __init__(self       ,
                 # --- Dimensions injected by PipelineBuilder ---
                 protein_emb_dim: dict,
                 ligand_emb_dim: dict,
                 out_dim: int,
                 # --- Hyperparameters from config ---
                 n_heads    ,       # number of attention heads
                 d_ff       ,       # hidden size of MLP
                ):
        super().__init__()

        d_model = protein_emb_dim['embedding'][1]
        cmpd_dim = ligand_emb_dim['embedding'][0]

        self.cmpd_proj = nn.Linear(cmpd_dim, d_model)           # [B, cmpd_dim] -> [B, d_model]

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = d_model ,
            num_heads   = n_heads ,
            batch_first = True    ,                             # expects [B, L, d_model]
        )
     
        #self.out_proj = nn.Linear(d_model, out_dim)            # [B, 1, d_model] -> [B, 1, out_dim]

        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff)   ,   # [B, out_dim] -> [B, d_ff]
                                 nn.ReLU(inplace=True)      ,
                                 nn.Linear(d_ff, out_dim) ,     # [B, d_ff] -> [B, out_dim]
        )

    def forward(self, protein_embedding, ligand_embedding, protein_mask=None, **kwargs):
        """
        protein_embedding : [bs, seqlen, d_model ]
        ligand_embedding  : [bs, cmpd_dim        ]
        protein_mask      : [bs, seqlen          ] # 0 = padding, 1 = valid (invert for key_padding_mask)
        """

        #print("protein_mask: ", protein_mask)

        # q, k, v
        q = self.cmpd_proj(ligand_embedding)                    # [bs, d_model]
        q = q.unsqueeze(1)                                      # [bs, 1, d_model]
        k = v = protein_embedding                               # [bs, seqlen, d_model]

        # mask
        key_padding_mask = None
        if protein_mask is not None:
            key_padding_mask = (protein_mask == 0)              # [B, L] -> bool

        # attention
        attn_out, _ = self.cross_attn(
            query = q ,                                         # [B, 1, d_model]
            key   = k ,                                         # [B, seqlen, d_model]
            value = v ,                                         # [B, seqlen, d_model]
            key_padding_mask=key_padding_mask                   # [B, L]
        )                                                       # attn_out: [B, 1, d_model]

        # att out
        attn_out = attn_out.squeeze(1)                          # [B, d_model]

        # MLP
        #attn_out = self.out_proj(attn_out)                     # [B, out_dim]

        output   = self.mlp(attn_out)                           # [B, num_preds]

        return output                                           # predictions