from unittest.mock import patch, MagicMock
from unittest import mock
from src.datasets.base.processor import Processor
import pytest


@patch("src.common.services.verification.verify_args")
def test_processor_validate_args(mock_verify_args):
    mock_tokenizer_info = MagicMock()
    mock_tokenizer_info.__getitem__.side_effect = lambda key: None
    processor = Processor(mock_tokenizer_info)
    processor.validate_args(some_arg="some_value")
    mock_verify_args.call_count == 1
    mock_verify_args.call_args == mock.call({}, some_arg="some_value")


def test_processor_process():
    mock_tokenizer_info = MagicMock()
    mock_tokenizer_info.__getitem__.side_effect = lambda key: None
    processor = Processor(mock_tokenizer_info)
    with pytest.raises(NotImplementedError):
        processor.process()


def test_processor_encode():
    mock_tokenizer_info = MagicMock()
    mock_tokenizer_info.__getitem__.side_effect = lambda key: None
    processor = Processor(mock_tokenizer_info)
    with pytest.raises(NotImplementedError):
        processor.encode("some_sample")
