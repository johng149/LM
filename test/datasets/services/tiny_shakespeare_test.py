from src.datasets.services.tiny_shakespeare import TinyShakespeareProcessor as Processor
from unittest.mock import patch, MagicMock, call
from src.tokenizers.services.available_tokenizers import available_tokenizers
from torch import Tensor
import torch
from src.common.models.dataloader_type import DataloaderType
from src.common.models.args_info import ArgInfo
from src.common.models.param_level import ParamLevel


def setup_tokenizer():
    tokenizer_info = available_tokenizers["gpt2"]
    return tokenizer_info()


def test_tiny_shakespeare_encode():
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
def test_tiny_shakespeare_helper(mock_load_dataset):
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


def test_tiny_shakespeare_collate_causal():
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


def test_tiny_shakespeare_collate_causal_same_len():
    tokenizer_info = setup_tokenizer()
    processor = Processor(tokenizer_info)
    collate_fn = processor.collate_causal_fn()
    batch = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    bos_idx = 0
    eos_idx = -1
    pad_idx = -2
    expected_source = torch.tensor(
        [
            [bos_idx, 1, 2, 3],
            [bos_idx, 4, 5, 6],
        ]
    )
    expected_target = torch.tensor(
        [
            [1, 2, 3, eos_idx],
            [4, 5, 6, eos_idx],
        ]
    )
    source, target = collate_fn(batch, bos_idx, eos_idx, pad_idx)
    assert torch.equal(source, expected_source)
    assert torch.equal(target, expected_target)


@patch("src.datasets.services.tiny_shakespeare.load_from_disk")
def test_tiny_shakespeare_causal(mock_load_from_disk):
    max_length = 10
    mock_raw = {"text_encoded": [torch.arange(max_length).tolist()]}
    batch_size = 2
    kwargs = {
        "batch_size": batch_size,
        "max_length": max_length,
    }
    mock_load_from_disk.return_value = mock_raw
    tokenizer_info = setup_tokenizer()
    some_dataset_path = "some/path"
    processor = Processor(tokenizer_info)
    t = DataloaderType.TRAIN
    dataloader = processor.causal(some_dataset_path, t, **kwargs)
    assert dataloader is not None

    sample = next(iter(dataloader))
    source, target = sample[:-1], sample[-1]

    assert isinstance(source, list) or isinstance(source, tuple)
    assert len(source) == 1

    source = source[0]

    assert isinstance(target, Tensor)
    assert isinstance(source, Tensor)

    # since the dataloader passes in an index to the dataset to
    # produce a sequence, and we don't have access to the sequence
    # we can't say for certain what the output will be, however,
    # we can test for properties that should hold true
    assert source.shape[0] == kwargs["batch_size"]
    assert target.shape[0] == kwargs["batch_size"]
    assert source.shape[1] == target.shape[1]
    assert source.shape[1] >= 1

    # we check that across the sequence dimension, tensors
    # are monotonic increasing (except for special tokens)
    pad_idx = tokenizer_info.pad_idx
    bos_idx = tokenizer_info.bos_idx
    eos_idx = tokenizer_info.eos_idx
    shorted_source = source[:, :-1]
    shorted_target = target[:, :-1]
    source_is_pad = shorted_source == pad_idx
    source_is_bos = shorted_source == bos_idx
    source_is_eos = shorted_source == eos_idx
    target_is_pad = shorted_target == pad_idx
    target_is_bos = shorted_target == bos_idx
    target_is_eos = shorted_target == eos_idx
    source_diff = torch.diff(source) >= 0
    target_diff = torch.diff(target) >= 0
    assert torch.all(source_is_pad | source_is_bos | source_is_eos | source_diff)
    assert torch.all(target_is_pad | target_is_bos | target_is_eos | target_diff)


@patch("src.datasets.services.tiny_shakespeare.verify_args")
def test_tiny_shakespeare_causal_verify_args(mock_verify_args):
    tokenizer_info = setup_tokenizer()
    processor = Processor(tokenizer_info)
    args = {
        "some_arg": "some_value",
        "some_other_arg": 3,
        "some_bool": True,
    }
    expected_argsInfo = {
        "batch_size": ArgInfo(
            level=ParamLevel.REQUIRED,
            type=int,
            description="Batch size for the dataloader",
        ),
        "max_length": ArgInfo(
            level=ParamLevel.REQUIRED,
            type=int,
            description="Max length of sequence used to train the model",
        ),
    }
    processor.causal_verify_args(**args)
    assert mock_verify_args.call_count == 1
    assert mock_verify_args.call_args == call(expected_argsInfo, **args)
