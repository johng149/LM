import torch
from src.nn.utils.dag_masking import acyclic_mask
from src.nn.utils.dag_masking import padding_transition_mask
from src.nn.utils.dag_masking import masking_helper
from src.nn.utils.dag_masking import masking


def test_acyclic_mask():
    vertices = 3
    device = "cpu"
    mask = acyclic_mask(vertices, device)
    expected = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]], device=device, dtype=torch.float32
    )
    assert torch.all(mask == expected)


def test_padding_transition_mask():
    batch_size = 2
    vertices = 4
    vertex_lens = torch.tensor([2, 3])
    device = "cpu"
    mask = padding_transition_mask(batch_size, vertices, vertex_lens, device)
    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    assert torch.all(mask == expected)


def test_masking_helper():
    batch_size = 2
    vertices = 4
    vertex_lens = torch.tensor([2, 3])
    device = "cpu"

    a_mask = acyclic_mask(vertices, device)
    p_mask = padding_transition_mask(batch_size, vertices, vertex_lens, device)
    mask = masking_helper(batch_size, vertices, vertex_lens, device)
    expected = p_mask + a_mask
    assert torch.all(mask == expected)


def test_masking():
    batch_size = 2
    vertices = 4
    vertex_lens = torch.tensor([2, 3])
    device = "cpu"

    mask, row_mask = masking(batch_size, vertices, vertex_lens, device)

    expected_mask = torch.tensor(
        [
            [
                [False, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ],
            [
                [False, True, True, False],
                [False, False, True, False],
                [True, True, True, True],
                [True, True, True, True],
            ],
        ]
    )
    expected_row_mask = torch.tensor(
        [[[False], [True], [True], [True]], [[False], [False], [True], [True]]]
    )
