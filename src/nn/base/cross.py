import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention as sdpa
from .phm import phm


class XAttention(nn.Module):
    """
    query has access to all the keys and values,
    might still want to use attn_mask to prevent attending to padding tokens
    """

    def __init__(self, embedding_dim, num_heads, factor, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.dropout_prob = dropout
        self.q_attn_proj = phm(factor, embedding_dim, embedding_dim)
        self.kv_attn_proj = phm(factor, embedding_dim, embedding_dim * 2)
        self.out_proj = phm(factor, embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, q_x, kv_x, mask=None):
        batch_size, seq_len, embedding_dim = q_x.shape
        q = (
            self.q_attn_proj(q_x)
            .view(batch_size, -1, self.num_heads, embedding_dim // self.num_heads)
            .transpose(1, 2)
        )
        k, v = self.kv_attn_proj(kv_x).split(embedding_dim, dim=-1)
        k = k.view(
            batch_size, -1, self.num_heads, embedding_dim // self.num_heads
        ).transpose(1, 2)
        v = v.view(
            batch_size, -1, self.num_heads, embedding_dim // self.num_heads
        ).transpose(1, 2)
        attn = sdpa(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(batch_size, -1, embedding_dim)
        attn = self.attn_dropout(attn)
        return self.out_dropout(self.out_proj(attn))


class XBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.dropout_prob = dropout
        self.attn = XAttention(embedding_dim, num_heads, factor)
        self.ffn = phm(factor, embedding_dim, embedding_dim * 4)
        self.ffn2 = phm(factor, embedding_dim * 4, embedding_dim)
        self.norm1q = nn.LayerNorm(embedding_dim)
        self.norm1kv = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn_activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn2_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_x, mask=None):
        attn = self.attn(self.norm1q(x), self.norm1kv(kv_x), mask=mask)
        x = x + attn
        ffn = self.ffn_activation(self.ffn(self.norm2(x)))
        ffn = self.ffn_dropout(ffn)
        ffn = self.ffn2(ffn)
        ffn = self.ffn2_dropout(ffn)
        x = x + ffn
        return x
