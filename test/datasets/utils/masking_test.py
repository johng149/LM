from src.datasets.utils.masking import causal_mask
import torch


def test_causal_mask_no_padding():
    input_tensor = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 1],
        ]
    )
    pad_idx = 0
    mask = causal_mask(input_tensor, pad_idx)
    expected = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ],
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ],
        ]
    )
    assert torch.equal(mask, expected)
