
# MIT License

# Copyright (c) 2024 D. Carpintero
# Modifications Copyright (c)  Da Saem Lee, 2025
import torch
from torch import nn

from layers.multihead_diffattn import MultiheadDiffAttn

class AttentionHead(nn.Module):
    def __init__(self, n_embd, n_headd):
        super().__init__()
        self.qkv = nn.Linear(n_embd, n_headd * 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, 0.0)

        return torch.bmm(torch.softmax(attn_scores, axis=-1), v)

    def forward(self, x, mask):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.scaled_dot_product_attention(q, k, v, mask=mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.heads = nn.ModuleList([AttentionHead(n_embd, n_embd // n_head) for _ in range(n_head)])
        self.output_linear = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask):
        return self.output_linear(torch.cat([head(x, mask) for head in self.heads], dim=-1))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_embd, ff_dim):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(n_embd, ff_dim), 
                                nn.GELU(), 
                                nn.Dropout(0.1),
                                nn.Linear(ff_dim, n_embd))

    def forward(self, x):
        return self.ff(x)

class TransformerDecoderLayerDiff(nn.Module):
    def __init__(self, n_embd, n_head, ff_dim, layer_id=0, dropout=0.1):
        super().__init__()
        
        self.a2a_mha = MultiHeadAttention(n_embd, n_head)
        self.map2a_diff = MultiheadDiffAttn(n_embd, layer_id, n_head)
        
        self.norm_1 = nn.LayerNorm(n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)
        self.norm_3 = nn.LayerNorm(n_embd)
        
        self.feed_forward = PositionWiseFeedForward(n_embd, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, map_enc, map_mask=None, mask=None):
        attn, attn_weights = self.map2a_diff(x, map_enc, attn_mask=map_mask)
        x = self.norm_1(x + self.dropout(attn))
        x = self.norm_2(x + self.dropout(self.a2a_mha(x, mask)))

        return self.norm_3(x + self.feed_forward(x))

