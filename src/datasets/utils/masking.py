import torch
from torch import Tensor
from typing import Tuple, List


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
    Combine multiple masks into a single mask.

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


def multiheadify_sdpa_helper(mask: Tensor) -> Tensor:
    """
    Attempt to make given mask compatible with multihead scaled dot product attention.
    This is done by unsqueezing the mask such that it goes from
    (batch_size, seq_len1, seq_len2) to (batch_size, 1, seq_len1, seq_len2).
    @param mask: a tensor of shape (batch_size, seq_len, seq_len)
    @return: a tensor of shape (batch_size, 1, seq_len, seq_len)
    """
    return mask.unsqueeze(1)


def multiheadify(x: List[Tensor | None], num_heads: int):
    """
    Given a list of masks, check if they are compatible with multihead attention.
    If not, make them compatible by using multiheadify_sdpa_helper.
    @param x: a list of tensors of varying shapes
    @param num_heads: the number of heads in the multihead attention
    @return: a list of tensors of varying shapes
    """
    return [
        (
            multiheadify_sdpa_helper(mask)
            if mask is not None and mask.dim() == 3 and num_heads > 1
            else mask
        )
        for mask in x
    ]
