from unittest.mock import patch, MagicMock
from unittest import mock
from src.datasets.base.processor import Processor
import pytest
import json


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


def test_processor_process_helper():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    with pytest.raises(NotImplementedError):
        processor.process_helper("some_save_path")


@patch.object(Processor, "already_cached")
@patch.object(Processor, "process_helper")
@patch.object(Processor, "set_cache")
def test_processor_process(mock_already_cached, mock_process_helper, mock_set_cache):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    processor.already_cached.return_value = False

    save_path = "some_save_path"
    processor.process(save_path)

    assert processor.already_cached.call_count == 1
    assert processor.already_cached.call_args == mock.call(
        save_path, Processor.__name__
    )
    assert processor.process_helper.call_count == 1
    assert processor.process_helper.call_args == mock.call(save_path)
    assert processor.set_cache.call_count == 1
    assert processor.set_cache.call_args == mock.call(save_path, Processor.__name__)


@patch.object(Processor, "already_cached")
@patch.object(Processor, "process_helper")
@patch.object(Processor, "set_cache")
def test_processor_process2(mock_already_cached, mock_process_helper, mock_set_cache):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    processor.already_cached.return_value = True

    save_path = "some_save_path"
    processor.process(save_path)

    assert processor.already_cached.call_count == 1
    assert processor.already_cached.call_args == mock.call(
        save_path, Processor.__name__
    )
    assert processor.process_helper.call_count == 0
    assert processor.set_cache.call_count == 0
