import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2))
        scores.masked_fill_(attn_mask,-1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttentionwithonekey(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = out_dim
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc  = nn.Linear(n_heads * d_v, out_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        Q = self.W_Q(input_Q).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(input_V.size(0),-1, self.n_heads, self.d_v).transpose(1,2)
        #print(Q.size(), K.size())
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(input_Q.size(0), -1, self.n_heads * self.d_v)
        output = self.fc(context) # [batch_size, len_q, out_dim]
        return output, attn


class PoswiseFeedForwardNet_SAtt(nn.Module):
    def __init__(self, d_model, cmpd_dim, d_ff, num_preds=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model + cmpd_dim, d_ff) ,
            nn.ReLU()                           ,
            nn.Linear(d_ff, num_preds)                  ,
            )

    def forward(self, inputs, compound):
        '''
        inputs: [batch_size, src_len, out_dim]
        '''
        inputs = torch.cat((torch.flatten(inputs, start_dim = 1), compound), 1)
        output = self.fc(inputs)
        return output


class EncoderLayer_SAtt(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, out_dim, cmpd_dim, d_ff): #out_dim = 1, n_head = 4, d_k = 256
        super(EncoderLayer_SAtt, self).__init__()
        self.emb_self_attn = MultiHeadAttentionwithonekey(d_model,d_k,n_heads,d_v,out_dim)
        self.pos_ffn = PoswiseFeedForwardNet_SAtt(d_model, cmpd_dim, d_ff)

    def forward(self, input_emb, emb_self_attn_mask, input_mask, compound):
        '''
        input_emb: [batch_size, src_len, d_model]
        emb_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # output_emb: [batch_size, src_len, 1], attn: [batch_size, n_heads, src_len, src_len]
        output_emb, attn = self.emb_self_attn(input_emb, input_emb, input_emb, emb_self_attn_mask) # input_emb to same Q,K,V
        batch_mask  = input_mask.unsqueeze(2)
        output_emb  = output_emb*batch_mask
        pos_weights = nn.Softmax(dim=1)(output_emb.masked_fill_(input_mask.unsqueeze(2).data.eq(0), -1e9)).permute(0,2,1) # [ batch_size, 1, src_len]
        output_emb  = torch.matmul(pos_weights, input_emb)
        output_emb  = self.pos_ffn(output_emb, compound) # output_emb: [batch_size, d_model]
        return output_emb, pos_weights