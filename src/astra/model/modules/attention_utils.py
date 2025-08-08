import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1)**0.5)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, out_dim, bias=False)
        self.attention = ScaledDotProductAttention()

    def forward(self, input_Q, input_K, input_V, attn_mask):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.attention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output, attn

class PoswiseFeedForwardNet_SAtt(nn.Module):
    def __init__(self, attn_out_dim, d_model, cmpd_dim, d_ff, num_preds):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(attn_out_dim*d_model + cmpd_dim, d_ff), # TODO: d_model+cmpd_dim needs to become attn_dim*prot_dim (or see other solution in forward())
            nn.ReLU(),
            nn.Linear(d_ff, num_preds),
        )

    def forward(self, pooled_protein_emb, compound):
        # Concatenate the fixed-size protein vector with the ligand vector
        combined_input = torch.cat((torch.flatten(pooled_protein_emb, start_dim=1), compound), 1) # TODO: pooled_protein_emb.shape = [batch_size, attn_dim, prot_dim] right now (which leads to mismatch error when running self.fc())
                                                                                                  # So we need to decide whether to make it prot_dim (aka d_model in this case) or attn_dim*prot_dim (which could be HUGE, e.g. 32*320=10240)
        output = self.fc(combined_input)
        return output
    

class EncoderLayer_SAtt(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, attn_out_dim, cmpd_dim, d_ff, final_out_dim):
        super().__init__()
        self.emb_self_attn = MultiHeadAttention(d_model, d_k, n_heads, d_v, attn_out_dim)
        self.pos_ffn = PoswiseFeedForwardNet_SAtt(attn_out_dim, d_model, cmpd_dim, d_ff, num_preds=final_out_dim)

    def forward(self, input_emb, self_attn_mask, input_pad_mask, compound):
        # 1. Apply multi-head self-attention to the protein embedding
        # output_emb: [batch_size, src_len, attn_out_dim]
        # attn: [batch_size, n_heads, src_len, src_len]
        output_emb, attn = self.emb_self_attn(input_emb, input_emb, input_emb, self_attn_mask)
        
        # 2. Use attention output as weights to create a weighted-average of the original protein embedding
        batch_mask = input_pad_mask.unsqueeze(2)
        output_emb = output_emb * batch_mask # Zero out pads

        # Softmax over sequence length to get pooling weights
        # [batch_size, 1, src_len]
        pos_weights = nn.Softmax(dim=1)(output_emb.masked_fill(batch_mask.data.eq(0), -1e9)).permute(0, 2, 1)
        
        # Apply weights to the original embedding to get a fixed-size representation
        # [batch_size, 1, d_model]
        pooled_protein_emb = torch.matmul(pos_weights, input_emb)

        # 3. Pass the pooled protein vector and ligand vector to the final FFN
        # output: [batch_size, final_out_dim]
        output = self.pos_ffn(pooled_protein_emb, compound)
        return output