from unittest.mock import patch, MagicMock
from unittest import mock
from src.datasets.base.processor import Processor
import pytest
import json
from pathlib import Path
from src.common.models.dataloader_type import DataloaderType
from pyfakefs.fake_filesystem import FakeFilesystem


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


@patch("src.datasets.base.processor.verify_args")
def test_processor_process_verify_args(mock_verify_args):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    mock_verify_args.return_value = None
    assert processor.process_verify_args() is None
    assert mock_verify_args.call_count == 1
    assert mock_verify_args.call_args == mock.call({}, **{})


def test_processor_format_cache_path():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)

    path = "some_path"
    class_name = "some_class"

    expected_path = Path(path) / f"{class_name}.json"
    result = processor.format_cache_path(path, class_name)
    assert result == expected_path


def test_processor_already_cached_miss_no_file(fs: FakeFilesystem):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)

    path = "some_path"
    class_name = "some_class"
    kwargs = {"some": "kwargs"}

    result = processor.already_cached(path, class_name, **kwargs)
    assert not result


def test_processor_already_cached_miss_incorrect_kwargs(fs: FakeFilesystem):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)

    path = "some_path"
    class_name = "some_class"
    kwargs = {"some": "kwargs"}

    file_path = processor.format_cache_path(path, class_name)
    fs.create_file(str(file_path), contents=json.dumps({"some": "other_kwargs"}))

    result = processor.already_cached(path, class_name, **kwargs)
    assert not result


def test_processor_already_cached_hit(fs: FakeFilesystem):
    mock_tokenizer_info = MagicMock()
    mock_tokenizer_info.tokenizer_name = "some_tokenizer_name"
    mock_tokenizer_info.pad_idx = 0
    mock_tokenizer_info.bos_idx = 1
    mock_tokenizer_info.eos_idx = 2
    mock_tokenizer_info.mask_idx = 3
    mock_tokenizer_info.vocab_size = 4
    # set mock_tokenizer_info's properties
    processor = Processor(mock_tokenizer_info)

    path = "some_path"
    class_name = "some_class"
    kwargs = {"some": "kwargs"}
    saved_kwargs = {
        "kwargs": kwargs,
        "tokenizer_info": {
            "tokenizer_name": mock_tokenizer_info.tokenizer_name,
            "pad_idx": mock_tokenizer_info.pad_idx,
            "bos_idx": mock_tokenizer_info.bos_idx,
            "eos_idx": mock_tokenizer_info.eos_idx,
            "mask_idx": mock_tokenizer_info.mask_idx,
            "vocab_size": mock_tokenizer_info.vocab_size,
        },
    }

    file_path = processor.format_cache_path(path, class_name)
    fs.create_file(str(file_path), contents=json.dumps(saved_kwargs))

    result = processor.already_cached(path, class_name, **kwargs)
    assert result


def test_processor_already_cached_hit2(fs: FakeFilesystem):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)

    path = "some_path"
    class_name = "some_class"
    kwargs = {
        "some": "kwargs",
        "some_other": "kwargs",
        "some_bool": True,
        "some_int": 1,
        "some_float": 1.0,
        "some_list": [1, 2, 3],
        "some_dict": {"some": "dict"},
        "some_dict_mixed": {"some": "dict", "some_list": [1, 2, 3]},
        "some_null": None,
    }

    file_path = processor.format_cache_path(path, class_name)
    fs.create_file(str(file_path), contents=json.dumps(kwargs))

    result = processor.already_cached(path, class_name, **kwargs)
    assert result


def test_processor_already_cached_hit2(fs: FakeFilesystem):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)

    path = "some_path"
    class_name = "some_class"
    kwargs = {
        "some": "kwargs",
        "some_other": "kwargs",
        "some_bool": True,
        "some_int": 1,
        "some_float": 1.0,
        "some_list": [1, 2, 3],
        "some_dict": {"some": "dict"},
        "some_dict_mixed": {"some": "dict", "some_list": [1, 2, 3]},
        "some_null": None,
    }

    kwargs_saved = {
        "some": "kwargs",
        "some_other": "kwargs",
        "some_bool": True,
        "some_int": 1,
        "some_float": 1.0,
        "some_list": [1, 2, 3],
        "some_dict": {"some": "dict"},
        "some_dict_mixed": {"what": "dict", "some_list": [1, 2, 3]},
        "some_null": None,
    }

    file_path = processor.format_cache_path(path, class_name)
    fs.create_file(str(file_path), contents=json.dumps(kwargs_saved))

    result = processor.already_cached(path, class_name, **kwargs)
    assert not result


def test_processor_set_cache(fs: FakeFilesystem):
    mock_tokenizer_info = MagicMock()
    mock_tokenizer_info.tokenizer_name = "some_tokenizer_name"
    mock_tokenizer_info.pad_idx = 0
    mock_tokenizer_info.bos_idx = 1
    mock_tokenizer_info.eos_idx = 2
    mock_tokenizer_info.mask_idx = 3
    mock_tokenizer_info.vocab_size = 4
    processor = Processor(mock_tokenizer_info)

    path = "some_path"
    class_name = "some_class"
    kwargs = {"some": "kwargs"}
    expected_loaded = {
        "kwargs": kwargs,
        "tokenizer_info": {
            "tokenizer_name": mock_tokenizer_info.tokenizer_name,
            "pad_idx": mock_tokenizer_info.pad_idx,
            "bos_idx": mock_tokenizer_info.bos_idx,
            "eos_idx": mock_tokenizer_info.eos_idx,
            "mask_idx": mock_tokenizer_info.mask_idx,
            "vocab_size": mock_tokenizer_info.vocab_size,
        },
    }

    fs.create_dir(path)

    processor.set_cache(path, class_name, **kwargs)

    file_path = processor.format_cache_path(path, class_name)
    assert file_path.exists()

    with open(file_path, "r") as f:
        saved_kwargs = json.load(f)
        assert saved_kwargs == expected_loaded


def test_processor_encode():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    with pytest.raises(NotImplementedError):
        processor.encode("some_sample")


def test_processor_seq2seq():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert processor.seq2seq("some_dataset_path", DataloaderType.TRAIN) is None


@patch("src.datasets.base.processor.verify_args")
def test_processor_seq2seq_verify_args(mock_verify_args):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    mock_verify_args.return_value = None
    assert processor.seq2seq_verify_args() is None
    assert mock_verify_args.call_count == 1
    assert mock_verify_args.call_args == mock.call({}, **{})


def test_processor_collate_seq2seq_fn():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert processor.collate_seq2seq_fn() is None


def test_processor_supports_seq2seq():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert not processor.supports_seq2seq()


def test_processor_causal():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert (
        processor.causal("some_dataset_path", DataloaderType.TRAIN, batch_size=2)
        is None
    )


@patch("src.datasets.base.processor.verify_args")
def test_processor_causal_verify_args(mock_verify_args):
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    mock_verify_args.return_value = None
    assert processor.causal_verify_args() is None
    assert mock_verify_args.call_count == 1
    assert mock_verify_args.call_args == mock.call({}, **{})


def test_processor_collate_causal_fn():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert processor.collate_causal_fn() is None


def test_processor_supports_causal():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    assert not processor.supports_causal()


def test_processor_format_dataset_path_train():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    dataset_path = "some_dataset_path"
    result = processor.format_dataset_path(dataset_path, DataloaderType.TRAIN)
    assert result == Path(dataset_path) / "train"


def test_processor_format_dataset_path_val():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    dataset_path = "some_dataset_path"
    result = processor.format_dataset_path(dataset_path, DataloaderType.VALIDATION)
    assert result == Path(dataset_path) / "val"


def test_processor_format_dataset_path_unknown():
    mock_tokenizer_info = MagicMock()
    processor = Processor(mock_tokenizer_info)
    dataset_path = "some_dataset_path"
    with pytest.raises(ValueError):
        processor.format_dataset_path(dataset_path, "unknown_type")
