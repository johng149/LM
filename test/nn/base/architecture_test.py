from src.nn.base.architecture import Architecture
import pytest
from unittest.mock import MagicMock
import torch


def test_architecture():
    arch = Architecture()
    with pytest.raises(NotImplementedError):
        arch.init_kwargs()


def test_architecture_naive_decode_batch_fail():
    arch = Architecture()
    batch, seq, max_len = 2, 10, 20
    x = torch.randn(batch, seq)
    strat = MagicMock()
    with pytest.raises(AssertionError):
        arch.naive_inference(x, strat, max_len)


def test_architecture_naive_decode_padding_fail():
    arch = Architecture()
    batch, seq, max_len = 1, 10, 20
    x = torch.randint(0, 100, (batch, seq))
    strat = MagicMock()
    strat.pad_id.return_value = 0
    x[0][-1] = strat.pad_id()
    with pytest.raises(AssertionError):
        arch.naive_inference(x, strat, max_len)


def test_architecture_naive_decode_not_implemented():
    arch = Architecture()
    max_len = 20
    x = torch.tensor([[1, 2, 3]])
    strat = MagicMock()
    strat.pad_id.return_value = 0
    assert arch.naive_inference(x, strat, max_len) == None
