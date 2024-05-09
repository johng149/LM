from src.nn.base.transformer import TransformerAttention, TransformerBlock
import torch
from torch.nn import Linear
import pytest


def test_TransformerAttention():
    emb_dim = 4
    num_heads = 2
    factor = -1
    attn = TransformerAttention(emb_dim, num_heads, factor)
    assert isinstance(attn.attn_proj, Linear)
    assert isinstance(attn.out_proj, Linear)

    # for reproducibility, we set the weights to 1.0
    attn.attn_proj.weight.data.fill_(1.0)
    attn.out_proj.weight.data.fill_(1.0)

    # and the bias to 0.0
    attn.attn_proj.bias.data.fill_(0.0)
    attn.out_proj.bias.data.fill_(0.0)

    batch_size = 3
    q_seq_len = 3
    kv_seq_len = 3

    q_x = torch.randn(batch_size, q_seq_len, emb_dim)

    kv_is_pad = torch.tensor([False, False, True])
    kv_is_pad2 = torch.tensor([True, False, True])
    kv_is_pad3 = torch.tensor([False, False, False])

    kv_x = torch.randn(batch_size, kv_seq_len, emb_dim)
    kv_x[0][kv_is_pad] = -10000
    kv_x[1][kv_is_pad2] = -10000
    kv_x[2][kv_is_pad3] = -10000
    kv_x[0][~kv_is_pad] = 0
    kv_x[1][~kv_is_pad2] = 0
    kv_x[2][~kv_is_pad3] = 0

    mask = [
        kv_is_pad.repeat(batch_size, 1),
        kv_is_pad2.repeat(batch_size, 1),
        kv_is_pad3.repeat(batch_size, 1),
    ]

    mask = torch.stack(mask, dim=0)
    mask = mask.unsqueeze(1)

    q, k, v = attn.split_heads(q_x, kv_x, kv_x)
    attn_out = attn.attn(q, k, v, ~mask)
    assert attn_out.shape == (batch_size, q_seq_len, emb_dim)

    # because of the way we set the input tensors, all the masked out values
    # are -10000 while the unmasked values are 0, and so the output should be
    # all zeros as well
    assert torch.allclose(attn_out, torch.zeros_like(attn_out))


def test_TransformerAttention_not_unsqueezed_mask():
    emb_dim = 4
    num_heads = 2
    factor = -1
    attn = TransformerAttention(emb_dim, num_heads, factor)
    assert isinstance(attn.attn_proj, Linear)
    assert isinstance(attn.out_proj, Linear)

    # for reproducibility, we set the weights to 1.0
    attn.attn_proj.weight.data.fill_(1.0)
    attn.out_proj.weight.data.fill_(1.0)

    # and the bias to 0.0
    attn.attn_proj.bias.data.fill_(0.0)
    attn.out_proj.bias.data.fill_(0.0)

    batch_size = 3
    q_seq_len = 3
    kv_seq_len = 3

    q_x = torch.randn(batch_size, q_seq_len, emb_dim)

    kv_is_pad = torch.tensor([False, False, True])
    kv_is_pad2 = torch.tensor([True, False, True])
    kv_is_pad3 = torch.tensor([False, False, False])

    kv_x = torch.randn(batch_size, kv_seq_len, emb_dim)
    kv_x[0][kv_is_pad] = -10000
    kv_x[1][kv_is_pad2] = -10000
    kv_x[2][kv_is_pad3] = -10000
    kv_x[0][~kv_is_pad] = 0
    kv_x[1][~kv_is_pad2] = 0
    kv_x[2][~kv_is_pad3] = 0

    mask = [
        kv_is_pad.repeat(batch_size, 1),
        kv_is_pad2.repeat(batch_size, 1),
        kv_is_pad3.repeat(batch_size, 1),
    ]

    mask = torch.stack(mask, dim=0)

    q, k, v = attn.split_heads(q_x, kv_x, kv_x)
    with pytest.raises(RuntimeError):
        attn_out = attn.attn(q, k, v, ~mask)
