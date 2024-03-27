from unittest.mock import patch, MagicMock
from unittest import mock
from src.datasets.base.processor import Processor
import pytest


def test_processor_init():
    mock_tokenizer_info = MagicMock()
    mock_tokenizer = "some_tokenizer"
    mock_tokenizer_info.tokenizer = mock_tokenizer
    pad_idx = 0
    bos_idx = 1
    eos_idx = 2
    mask_idx = 3
    vocab_size = 4
    mock_tokenizer_info.pad_idx = pad_idx
    mock_tokenizer_info.bos_idx = bos_idx
    mock_tokenizer_info.eos_idx = eos_idx
    mock_tokenizer_info.mask_idx = mask_idx
    mock_tokenizer_info.vocab_size = vocab_size
    processor = Processor(mock_tokenizer_info)
    assert processor.info == mock_tokenizer_info
    assert processor.tokenizer == mock_tokenizer
    assert processor.pad_idx == pad_idx
    assert processor.bos_idx == bos_idx
    assert processor.eos_idx == eos_idx
    assert processor.mask_idx == mask_idx
    assert processor.vocab_size == vocab_size


@patch("src.common.services.verification.verify_args")
def test_processor_validate_args(mock_verify_args):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    processor.validate_args(some_arg="some_value")
    mock_verify_args.call_count == 1
    mock_verify_args.call_args == mock.call({}, some_arg="some_value")


def test_processor_process():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    with pytest.raises(NotImplementedError):
        processor.process()


def test_processor_encode():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    with pytest.raises(NotImplementedError):
        processor.encode("some_sample")


def test_processor_seq2seq():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert processor.seq2seq() == None
    assert not processor.supports_seq2seq()


def test_processor_causal():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert processor.causal() == None
    assert not processor.supports_causal()
