from src.datasets.utils.masking import (
    process_tokens,
    self_attn_pad_mask,
    cross_attn_pad_mask,
    causal_self_attn_mask,
    sdpa_flip,
)
from torch import Tensor
import torch
import pytest


def test_process_tokens():
    tokens = torch.tensor([[1, 2, 3, -1, -1], [4, 5, -1, 6, 7], [1, 2, 3, 4, 5]])
    pad_idx = -1
    expected_lens = torch.tensor([3, 4, 5])
    expected_is_not_pad = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ]
    )
    token_lens, token_is_not_pad = process_tokens(tokens, pad_idx)
    assert torch.equal(token_lens, expected_lens)
    assert torch.equal(token_is_not_pad, expected_is_not_pad)


def test_self_attn_pad_mask():
    is_not_pad = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [True, False, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    expected_output = torch.tensor(
        [
            [[True, True, False], [True, True, False], [True, True, False]],
            [[True, False, False], [True, False, False], [True, False, False]],
            [[True, False, True], [True, False, True], [True, False, True]],
            [[False, False, True], [False, False, True], [False, False, True]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ]
    )
    output = self_attn_pad_mask(is_not_pad)
    assert torch.equal(output, expected_output)


def test_cross_attn_pad_mask():
    kv_is_not_pad = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, False, True, True],
        ]
    )
    q_is_not_pad = torch.tensor(
        [
            [True, False, False],
            [True, False, True],
            [False, False, True],
        ]
    )
    expected_output = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ],
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
            ],
            [
                [True, False, True, True],
                [True, False, True, True],
                [True, False, True, True],
            ],
        ]
    )
    output = cross_attn_pad_mask(kv_is_not_pad, q_is_not_pad)
    assert torch.equal(output, expected_output)


def test_causal_self_attn_mask():
    samples = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected_output = torch.tensor(
        [
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
        ]
    )
    output = causal_self_attn_mask(samples)
    assert torch.equal(output, expected_output)


def test_sdpa_flip():
    mask = torch.rand(3, 4, 4) > 0.5
    expected_output = ~mask
    output = sdpa_flip(mask)
    assert torch.equal(output, expected_output)
