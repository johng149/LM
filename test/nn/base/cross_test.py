from src.nn.base.cross import XAttention
from src.nn.base.cross import XBlock
import torch
from torch.nn import Linear


def test_XAttention():
    emb_dim = 4
    num_heads = 2
    factor = -1
    xattn = XAttention(emb_dim, num_heads, factor)
    assert isinstance(xattn.q_attn_proj, Linear)
    assert isinstance(xattn.kv_attn_proj, Linear)
    assert isinstance(xattn.out_proj, Linear)

    # for reproducibility, we set the weights to 1.0
    xattn.q_attn_proj.weight.data.fill_(1.0)
    xattn.kv_attn_proj.weight.data.fill_(1.0)
    xattn.out_proj.weight.data.fill_(1.0)

    # and the bias to 0.0
    xattn.q_attn_proj.bias.data.fill_(0.0)
    xattn.kv_attn_proj.bias.data.fill_(0.0)
    xattn.out_proj.bias.data.fill_(0.0)

    batch_size = 3
    q_seq_len = 3
    kv_seq_len = 4

    q_x = torch.randn(batch_size, q_seq_len, emb_dim)

    kv_is_pad = torch.tensor([False, False, False, True])
    kv_is_pad2 = torch.tensor([True, False, True, False])
    kv_is_pad3 = torch.tensor([False, False, False, False])

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

    masked_out = xattn(q_x, kv_x, ~mask)
    assert masked_out.shape == (batch_size, q_seq_len, emb_dim)

    # because of the way we set the input tensors, all the masked out values
    # are -10000 while the unmasked values are 0, and so the output should be
    # all zeros as well
    assert torch.allclose(masked_out, torch.zeros_like(masked_out))
