import torch
from torch.nn import Module
from src.nn.base.architecture import Architecture
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Callable, Union
from torch.utils.data import DataLoader
from torch import device
from tqdm.auto import tqdm


def move_to_device(data, device: Union[str, device]):
    assert isinstance(data, list) or isinstance(data, tuple)
    return [d.to(device) for d in data]


def save_checkpoint(
    model: Architecture,
    optimizer: Optimizer,
    path: str,
    global_step: Optional[int] = None,
    tensorboard_dir: Optional[str] = None,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model.kwargs,
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            tensorboard_dir: tensorboard_dir,
        },
        path,
    )


def train_step(
    model: Architecture,
    optimizer: Optimizer,
    loss_fn: Union[Callable, Module],
    x,
    y,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    save_every: Optional[int] = None,
) -> float:
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if writer is not None:
        writer.add_scalar("Train/Loss", loss.item(), global_step)
    if (
        checkpoint_path is not None
        and save_every is not None
        and global_step % save_every == 0
    ):
        save_checkpoint(model, optimizer, checkpoint_path, global_step)
    return loss.item()


def test_step(
    model: Architecture,
    loss_fn: Union[Callable, Module],
    x,
    y,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
) -> float:
    with torch.no_grad():
        output = model(x)
        loss = loss_fn(output, y)
        if writer is not None:
            writer.add_scalar("Test/Loss", loss.item(), global_step)
    return loss.item()


def train(
    model: Architecture,
    optimizer: Optimizer,
    loss_fn: Union[Callable, Module],
    seen_epochs: int,
    target_epochs: int,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    writer: Optional[SummaryWriter] = None,
    checkpoint_path: Optional[str] = None,
    save_every: Optional[int] = None,
    test_every: Optional[int] = None,
    device: Union[str, device] = "cpu",
):
    """
    Trains given model for epochs equal to the difference between target and seen epochs.

    @param model: The model to train
    @param optimizer: The optimizer to use
    @param loss_fn: The loss function to use
    @param seen_epochs: The number of epochs already trained
    @param target_epochs: The number of epochs to train for
    @param train_loader: The DataLoader for training
    @param test_loader: The DataLoader for testing
    @param writer: The SummaryWriter to log to
    @param checkpoint_path: The path to save checkpoints to
    @param save_every: Save a checkpoint every n epochs
    @param test_every: Test the model every n epochs
    @param device: The device to train on

    It is assumed that the dataloaders used returns tuples or lists of tensors,
    and that the last element of the tuple or list is the target used in loss function.
    It also assumes that model has already been moved to the specified device
    """
    epoch = seen_epochs
    try:
        train_iter = iter(train_loader)
        test_iter = iter(test_loader) if test_loader is not None else None
        for epoch in tqdm(range(seen_epochs, target_epochs)):
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data = next(train_iter)
            data = move_to_device(data, device)
            x, y = data[:-1], data[-1]
            loss = train_step(
                model,
                optimizer,
                loss_fn,
                x,
                y,
                writer,
                epoch,
                checkpoint_path,
                save_every,
            )
            if (
                test_loader is not None
                and test_every is not None
                and epoch % test_every == 0
            ):
                try:
                    data = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    data = next(test_iter)
                data = move_to_device(data, device)
                x, y = data[:-1], data[-1]
                test_loss = test_step(model, loss_fn, x, y, writer, epoch)
    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, checkpoint_path, epoch)
    finally:
        save_checkpoint(model, optimizer, checkpoint_path, epoch)
