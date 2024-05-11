import pytest
from src.nn.models.decoding_strat_model import DecodingStrategy
from unittest.mock import MagicMock


def test_decoding_strat_init():
    info = MagicMock()
    device = "cpu"
    info.pad_idx = 0
    info.eos_idx = 1
    strat = DecodingStrategy(info, device)
    assert strat.info == info
    assert strat.device == device
    assert strat.pad_id() == 0
    assert strat.eos_id() == 1
    with pytest.raises(NotImplementedError):
        strat.decode(MagicMock())
