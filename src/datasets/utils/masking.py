import torch
from torch import Tensor
from typing import Tuple


def process_tokens(tokens: Tensor, pad_idx: int) -> Tuple[Tensor, Tensor]:
    """
    Process tokens to create information needed for creating masks.

    @param tokens: the tokens to process, shape (batch_size, seq_len)
    @param pad_idx: the index of the padding token
    @returns: a tuple containing the number of tokens in each sequence
    that is not padding and a boolean tensor indicating which tokens
    are not padding, shape (batch_size, seq_len)
    """
    token_is_not_pad = tokens != pad_idx
    token_lens = torch.sum(token_is_not_pad, dim=1)
    return token_lens, token_is_not_pad


def self_attn_pad_mask(is_not_pad: Tensor) -> Tensor:
    """
    @param not_is_pad: bool tensor of shape (batch_size x seq_len)
        where True indicates a non-padding token
    @return: bool tensor of shape (batch_size x seq_len x seq_len)
        where the value at position (i, j, k) is True if the
        attention matrix value at that same position should NOT be masked
    """
    batch_size, l = is_not_pad.shape
    return is_not_pad.unsqueeze(-1).expand(-1, -1, l).transpose(-2, -1)


def cross_attn_pad_mask(kv_is_not_pad: Tensor, q_is_not_pad: Tensor):
    """
    @param not_is_pad_kv: bool tensor of shape (batch_size x seq_len_kv)
        where True indicates a non-padding token
    @param not_is_pad_q: bool tensor of shape (batch_size x seq_len_q)
        where True indicates a non-padding token
    @return: bool tensor of shape (batch_size x seq_len_q x seq_len_kv)
        where the value at position (i, j, k) is True if the attention
        matrix value at that same position should not be masked
    """
    batch_size, l_kv = kv_is_not_pad.shape
    _, l_q = q_is_not_pad.shape
    return kv_is_not_pad.unsqueeze(-1).expand(-1, -1, l_q).transpose(-2, -1)


def causal_self_attn_mask(samples: Tensor):
    """
    @param samples: a tensor of shape (batch_size, seq_len)
        containing the input data
    @return: a tensor of shape (batch_size, seq_len, seq_len)
        where the value at position (i, j, k) is True if the
        attention matrix value at that same position should not be masked
    """
    batch_size, l = samples.shape
    return (
        torch.tril(torch.ones((l, l), device=samples.device))
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .bool()
    )


# accepts arbitrary number of masks
def combine_masks_before_flip(*masks: Tensor) -> Tensor:
    """
    Combine multiple masks into a single mask. This should be
    used prior to the call to sdpa_flip, as it assumes that
    elements in the mask that are True are ones that should
    not be masked as it uses the logical AND operation to
    combine the masks.

    @param masks: a variable number of masks, each of shape
        (batch_size, seq_len, seq_len), there must be at least
        one mask
    @return: a tensor of shape (batch_size, seq_len, seq_len)
        where the value at position (i, j, k) is True if the
        attention matrix value at that same position should not be masked
    """
    assert len(masks) > 0
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = torch.logical_and(combined_mask, mask)
    return combined_mask


def sdpa_flip(mask: Tensor) -> Tensor:
    """
    @param mask: a tensor of shape (batch_size, seq_len, seq_len)
        where the value at position (i, j, k) is True if the
        attention matrix value at that same position should not be masked
    @return: a tensor of shape (batch_size, seq_len, seq_len)
        where the value at position (i, j, k) is False if the
        attention matrix value at that same position should not be masked

    The reason for this is that scaled dot product attention (sdpa) expects
    a mask where True values indicate positions to mask out, while the
    masks we create indicate positions to NOT mask out. This function
    flips the mask so that it can be used with sdpa.
    """
    return torch.logical_not(mask)
