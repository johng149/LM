from src.entry_point.services.entry import load, load_from_checkpoint
from unittest.mock import MagicMock, patch
from unittest import mock


@patch("src.entry_point.services.entry.model_type_to_processor_dataloader")
@patch("src.entry_point.services.entry.model_type_to_processor_supports")
@patch("src.entry_point.services.entry.model_type_to_processor_verify_args")
@patch("src.entry_point.services.entry.available_models")
@patch("src.entry_point.services.entry.available_datasets")
@patch("src.entry_point.services.entry.available_tokenizers")
@patch("src.entry_point.services.entry.available_loss_fns")
@patch("src.entry_point.services.entry.train")
@patch("src.entry_point.services.entry.SummaryWriter")
def test_load(
    mock_summary_writer,
    mock_train,
    mock_available_loss_fns,
    mock_available_tokenizers,
    mock_available_datasets,
    mock_available_models,
    mock_model_type_to_processor_verify_args,
    mock_model_type_to_processor_supports,
    mock_model_type_to_processor_dataloader,
):
    jd_values = {
        "loss_fn": "some_loss_fn",
        "tokenizer": "some_tokenizer",
        "dataset": "some_dataset",
        "dataset_path": "some_dataset_path",
        "dataset_process_params": {"some": "params"},
        "model_type": "some_model_type",
        "train_dl_params": {"some": "params", "batch_size": 1},
        "test_dl_params": {"some": "params", "batch_size": 2},
        "use_validation": True,
        "model_kwargs": {"some": "kwargs"},
        "optimizer": "some_optimizer",
        "optimizer_kwargs": {"some": "kwargs", "lr": 0.1},
        "summary_writer_dir": "some_dir",
        "checkpoint_path": "some_checkpoint_path",
    }

    jd = MagicMock()
    jd.__getitem__.side_effect = jd_values.__getitem__
    jd.__contains__.side_effect = jd_values.__contains__
    jd.get.side_effect = jd_values.get

    mock_loss_fn = MagicMock()
    mock_loss_fn_return_value = "some loss fn return value"
    mock_loss_fn.return_value = mock_loss_fn_return_value

    loss_fn_values = {
        jd_values["loss_fn"]: mock_loss_fn,
    }
    mock_available_loss_fns.__getitem__.side_effect = loss_fn_values.__getitem__
    mock_available_loss_fns.__contains__.side_effect = loss_fn_values.__contains__

    mock_tokenizer = MagicMock()
    mock_tokenizer_vocab_size = 69
    mock_tokenizer.vocab_size = mock_tokenizer_vocab_size
    mock_tokenizer_fn = MagicMock()
    mock_tokenizer_fn.return_value = mock_tokenizer
    available_tokenizers_values = {
        jd_values["tokenizer"]: mock_tokenizer_fn,
    }
    mock_available_tokenizers.__getitem__.side_effect = (
        available_tokenizers_values.__getitem__
    )
    mock_available_tokenizers.__contains__.side_effect = (
        available_tokenizers_values.__contains__
    )

    mock_dataset = MagicMock()
    mock_dataset.process_verify_args.return_value = (None, False)
    mock_dataset_fn = MagicMock()
    mock_dataset_fn.return_value = mock_dataset
    available_datasets_values = {
        jd_values["dataset"]: mock_dataset_fn,
    }
    mock_available_datasets.__getitem__.side_effect = (
        available_datasets_values.__getitem__
    )
    mock_available_datasets.__contains__.side_effect = (
        available_datasets_values.__contains__
    )

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model_fn = MagicMock()
    mock_model_fn.return_value = mock_model
    available_models_values = {
        jd_values["model_type"]: mock_model_fn,
    }
    mock_available_models.__getitem__.side_effect = available_models_values.__getitem__
    mock_available_models.__contains__.side_effect = (
        available_models_values.__contains__
    )

    mock_model_type_to_processor_supports.return_value = lambda: True

    mock_dataset_dl_args_validator = MagicMock()
    mock_dataset_dl_args_validator.return_value = (None, False)
    mock_model_type_to_processor_verify_args.return_value = (
        mock_dataset_dl_args_validator
    )

    mock_dl = MagicMock()
    mock_dl_fn = MagicMock()
    mock_dl_fn.return_value = mock_dl
    mock_model_type_to_processor_dataloader.return_value = mock_dl_fn

    mock_summary_writer_instance = MagicMock()
    mock_summary_writer.return_value = mock_summary_writer_instance

    mock_optim_class = MagicMock()
    mock_optim = MagicMock()
    mock_optim_class.return_value = mock_optim
    mock_optim_maker = MagicMock()
    mock_optim_maker.return_value = mock_optim_class

    device = "some device"
    seen_epochs = 420
    target_epochs = 42069
    checkpoint_path = jd_values["checkpoint_path"]
    save_every = 101
    test_every = 202
    load(
        jd=jd,
        device=device,
        seen_epochs=seen_epochs,
        target_epochs=target_epochs,
        checkpoint_path=checkpoint_path,
        save_every=save_every,
        test_every=test_every,
        optim_maker=mock_optim_maker,
    )

    assert mock_available_loss_fns.__getitem__.call_count == 1
    assert mock_available_loss_fns.__getitem__.call_args == mock.call(
        jd_values["loss_fn"]
    )
