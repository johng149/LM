from src.nn.base.architecture import Architecture
import pytest
from unittest.mock import MagicMock
import torch


def test_architecture():
    arch = Architecture()
    with pytest.raises(NotImplementedError):
        arch.init_kwargs()


def test_architecture_naive_decode_not_implemented():
    arch = Architecture()
    max_len = 20
    x = torch.tensor([[1, 2, 3]])
    strat = MagicMock()
    strat.pad_id.return_value = 0
    assert arch.naive_inference(strat=strat) is None
