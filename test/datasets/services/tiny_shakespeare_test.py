from src.datasets.services.tiny_shakespeare import TinyShakespeareProcessor as Processor
from unittest.mock import patch, MagicMock, call
from src.tokenizers.services.available_tokenizers import available_tokenizers
from torch import Tensor
import torch
from src.common.models.dataloader_type import DataloaderType


def setup_tokenizer():
    tokenizer_info = available_tokenizers["gpt2"]
    return tokenizer_info


def test_encode():
    tokenizer_info = setup_tokenizer()
    processor = Processor(tokenizer_info)
    sample = {"text": "Hello World"}
    encoded = processor.encode(sample)
    assert "text" in encoded
    assert "text_encoded" in encoded
    assert isinstance(encoded["text"], str)
    assert isinstance(encoded["text_encoded"], Tensor)
    assert encoded["text"] == sample["text"]
    assert encoded["text_encoded"].shape == (2,)


@patch("src.datasets.services.tiny_shakespeare.load_dataset")
def test_process_helper(mock_load_dataset):
    mock_load_dataset_result = MagicMock()
    mock_load_dataset.return_value = mock_load_dataset_result
    mock_load_dataset_result.map.return_value = mock_load_dataset_result
    tokenizer_info = setup_tokenizer()
    processor = Processor(tokenizer_info)
    save_path = "some/path"
    processor.process_helper(save_path)
    assert mock_load_dataset.call_count == 2
    assert mock_load_dataset.call_args_list == [
        call("tiny_shakespeare", split="train"),
        call("tiny_shakespeare", split="validation"),
    ]
    assert mock_load_dataset_result.map.call_count == 2
    assert mock_load_dataset_result.map.call_args_list == [
        call(processor.encode),
        call(processor.encode),
    ]
    assert mock_load_dataset_result.save_to_disk.call_count == 2
    assert mock_load_dataset_result.save_to_disk.call_args_list == [
        call(processor.format_dataset_path(save_path, DataloaderType.TRAIN)),
        call(processor.format_dataset_path(save_path, DataloaderType.VALIDATION)),
    ]


def test_process_collate_causal():
    tokenizer_info = setup_tokenizer()
    processor = Processor(tokenizer_info)
    collate_fn = processor.collate_causal_fn()
    batch = [
        [1, 2, 3],
        [4, 5, 6, 4],
    ]
    bos_idx = 0
    eos_idx = -1
    pad_idx = -2

    expected_source = torch.tensor([[bos_idx, 1, 2, 3, pad_idx], [bos_idx, 4, 5, 6, 4]])

    expected_target = torch.tensor([[1, 2, 3, eos_idx, pad_idx], [4, 5, 6, 4, eos_idx]])

    source, target = collate_fn(batch, bos_idx, eos_idx, pad_idx)
    assert torch.equal(source, expected_source)
    assert torch.equal(target, expected_target)
