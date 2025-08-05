import torch
import torch.nn as nn
from astra.model.modules.attention_utils import *

# Orignial class name from CPI-Pred's Z05X_All_Models.py: SQembSAtt_CPenc_Model
# TODO: Include github link once published

class CpiPredSelfAttnModel(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, out_dim, cmpd_dim, d_ff):
        super().__init__()
        self.layers = EncoderLayer_SAtt(d_model, d_k, n_heads, d_v, out_dim, cmpd_dim, d_ff)

    def get_attn_pad_mask(self, seq_mask):
        batch_size, len_q = seq_mask.size()
        _, len_k = seq_mask.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_mask.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)        

    def forward(self, protein_embedding=None, ligand_embedding=None, protein_mask=None, **kwargs):
        '''
        input_emb  : [batch_size, src_len, embedding_dim]
        input_mask : [batch_size, src_len]
        '''

        # Adapt inputs to CPI-Pred's code
        input_emb = protein_embedding
        input_mask = protein_mask
        compound = ligand_embedding

        if input_mask is None:
            raise ValueError("CpiPredSelfAttnModel requires a protein_mask.")

        emb_self_attn_mask = self.get_attn_pad_mask(input_mask) # [batch_size, src_len, src_len]

        # output_emb: [batch_size, src_len, out_dim], emb_self_attn: [batch_size, n_heads, src_len, src_len]
        output_emb, emb_self_attn = self.layers(input_emb, emb_self_attn_mask, input_mask, compound)
        return output_emb #, emb_self_attn
    



    