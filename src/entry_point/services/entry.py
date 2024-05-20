from src.nn.services.available_models import available_models
from src.datasets.utils.available_datasets import available_datasets
from src.tokenizers.services.available_tokenizers import available_tokenizers
from src.nn.services.loss_fns import available_loss_fns
import torch
from torch import device
from src.nn.services.train import train
import importlib
from src.nn.services.available_models import model_type_to_processor_dataloader
from src.nn.services.available_models import model_type_to_processor_supports
from src.nn.services.available_models import model_type_to_processor_verify_args
from src.common.models.dataloader_type import DataloaderType
from torch.utils.tensorboard import SummaryWriter
import json
import os
import argparse


def load(
    jd: dict,
    device: device | str,
    seen_epochs: int,
    target_epochs: int,
    checkpoint_path: str,
    save_every: int,
    test_every: int,
):

    loss_fn = jd["loss_fn"]

    if loss_fn not in available_loss_fns:
        raise ValueError(f"Invalid loss function: {loss_fn}")

    tokenizer = jd["tokenizer"]

    if tokenizer not in available_tokenizers:
        raise ValueError(f"Invalid tokenizer: {tokenizer}")

    tokenizer = available_tokenizers[tokenizer]()

    dataset = jd["dataset"]

    if dataset not in available_datasets:
        raise ValueError(f"Invalid dataset: {dataset}")

    dataset = available_datasets[dataset](tokenizer)

    dataset_path = jd["dataset_path"]

    dataset_process_params = jd["dataset_process_params"]

    verifications, hasError = dataset.process_verify_args(**dataset_process_params)
    if hasError:
        raise ValueError(f"Invalid dataset process args: {verifications}")
    dataset.process(dataset_path, **dataset_process_params)

    model_type = jd["model_type"]

    dataset_supports_model = model_type_to_processor_supports(dataset, model_type)()
    if not dataset_supports_model:
        raise ValueError(f"Model type {model_type} not supported by dataset {dataset}")

    dataset_dl_args_validator = model_type_to_processor_verify_args(dataset, model_type)

    train_dl_params = jd["train_dl_params"]
    test_dl_params = jd.get("test_dl_params", None)

    verifications, hasError = dataset_dl_args_validator(**train_dl_params)
    if hasError:
        raise ValueError(f"Invalid dataset train dataloader args: {verifications}")

    if test_dl_params is not None:
        verifications, hasError = dataset_dl_args_validator(**test_dl_params)
        if hasError:
            raise ValueError(f"Invalid dataset test dataloader args: {verifications}")

    if model_type not in available_models:
        raise ValueError(f"Invalid model type: {model_type}")

    use_validation = jd.get("use_validation", False)

    train_dl = model_type_to_processor_dataloader(dataset, model_type)(
        dataset_path=dataset_path, type=DataloaderType.TRAIN, **train_dl_params
    )

    if use_validation:
        test_dl = model_type_to_processor_dataloader(dataset, model_type)(
            dataset_path=dataset_path, type=DataloaderType.VALIDATION, **test_dl_params
        )
    else:
        test_dl = None

    model_kwargs = jd["model_kwargs"]
    model_kwargs["vocab_size"] = tokenizer.vocab_size

    optimizer = jd["optimizer"]

    optimizer_kwargs = jd["optimizer_kwargs"]

    summary_writer_dir = jd["summary_writer_dir"]

    writer = SummaryWriter(summary_writer_dir)

    checkpoint_path = jd.get("checkpoint_path", None)

    model = available_models[model_type](**model_kwargs)
    model = model.to(device)

    optim_class = getattr(importlib.import_module("torch.optim"), optimizer)
    optimizer = optim_class(model.parameters(), **optimizer_kwargs)

    l = available_loss_fns[loss_fn](tokenizer)

    return (
        model,
        optimizer,
        l,
        seen_epochs,
        target_epochs,
        train_dl,
        test_dl,
        writer,
        checkpoint_path,
        save_every,
        test_every,
        device,
    )


def load_from_checkpoint(
    jd: dict,
    checkpoint_path: str,
    device: str | device,
    target_epochs: int,
    save_every: int,
    test_every: int,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    loss_fn = jd["loss_fn"]

    if loss_fn not in available_loss_fns:
        raise ValueError(f"Invalid loss function: {loss_fn}")

    tokenizer = jd["tokenizer"]

    if tokenizer not in available_tokenizers:
        raise ValueError(f"Invalid tokenizer: {tokenizer}")

    tokenizer = available_tokenizers[tokenizer]()

    dataset = jd["dataset"]

    if dataset not in available_datasets:
        raise ValueError(f"Invalid dataset: {dataset}")

    dataset = available_datasets[dataset](tokenizer)

    dataset_path = jd["dataset_path"]

    dataset_process_params = jd["dataset_process_params"]

    model_type = jd["model_type"]

    dataset_supports_model = model_type_to_processor_supports(dataset, model_type)()
    if not dataset_supports_model:
        raise ValueError(f"Model type {model_type} not supported by dataset {dataset}")

    dataset_dl_args_validator = model_type_to_processor_verify_args(dataset, model_type)

    train_dl_params = jd["train_dl_params"]
    test_dl_params = jd.get("test_dl_params", None)

    verifications, hasError = dataset_dl_args_validator(**train_dl_params)
    if hasError:
        raise ValueError(f"Invalid dataset process args: {verifications}")

    if test_dl_params is not None:
        verifications, hasError = dataset_dl_args_validator(**test_dl_params)
        if hasError:
            raise ValueError(f"Invalid dataset process args: {verifications}")

    if model_type not in available_models:
        raise ValueError(f"Invalid model type: {model_type}")

    use_validation = jd.get("use_validation", False)

    train_dl = model_type_to_processor_dataloader(dataset, model_type)(
        dataset_path=dataset_path, type=DataloaderType.TRAIN, **train_dl_params
    )

    if use_validation:
        test_dl = model_type_to_processor_dataloader(dataset, model_type)(
            dataset_path=dataset_path, type=DataloaderType.VALIDATION, **test_dl_params
        )
    else:
        test_dl = None

    model_state_dict = checkpoint["model_state_dict"]
    model_kwargs = checkpoint["model_kwargs"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    seen_epochs = checkpoint["global_step"]
    summary_writer_dir = checkpoint["tensorboard_dir"]

    writer = SummaryWriter(summary_writer_dir)

    model = available_models[model_type](**model_kwargs)
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    optimizer = jd["optimizer"]
    optimizer_kwargs = jd["optimizer_kwargs"]

    optim_class = getattr(importlib.import_module("torch.optim"), optimizer)
    optimizer = optim_class(model.parameters(), **optimizer_kwargs)
    optimizer.load_state_dict(optimizer_state_dict)

    l = available_loss_fns[loss_fn](tokenizer)

    return (
        model,
        optimizer,
        l,
        seen_epochs,
        target_epochs,
        train_dl,
        test_dl,
        writer,
        checkpoint_path,
        save_every,
        test_every,
        device,
    )


def entry_helper(
    jd_path: str,
    save_every: int,
    test_every: int,
    device: str | device,
    target_epochs: int,
):
    with open(jd_path, "r") as f:
        jd = json.load(f)

    if "checkpoint_path" in jd and os.path.exists(jd["checkpoint_path"]):
        return load_from_checkpoint(
            jd=jd,
            checkpoint_path=jd["checkpoint_path"],
            device=device,
            target_epochs=target_epochs,
            save_every=save_every,
            test_every=test_every,
        )
    else:
        return load(
            jd=jd,
            device=device,
            seen_epochs=0,
            target_epochs=target_epochs,
            checkpoint_path=jd.get("checkpoint_path", None),
            save_every=save_every,
            test_every=test_every,
        )


def entry(
    jd_path: str,
    save_every: int,
    test_every: int,
    device: str | device,
    target_epochs: int,
):
    (
        model,
        optimizer,
        loss_fn,
        seen_epochs,
        target_epochs,
        train_dl,
        test_dl,
        writer,
        checkpoint_path,
        save_every,
        test_every,
        device,
    ) = entry_helper(jd_path, save_every, test_every, device, target_epochs)
    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        seen_epochs=seen_epochs,
        target_epochs=target_epochs,
        train_loader=train_dl,
        test_loader=test_dl,
        writer=writer,
        checkpoint_path=checkpoint_path,
        save_every=save_every,
        test_every=test_every,
        device=device,
    )
