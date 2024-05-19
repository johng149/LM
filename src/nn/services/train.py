import torch
from torch.nn import Module
from src.nn.base.architecture import Architecture
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Callable, Union
from torch.utils.data import DataLoader
from torch import device
from tqdm.auto import tqdm
from torch import Tensor
import os


def ensure_dir_exists(path: str):
    """
    Given a path to a directory or file, ensure that the
    directory / parent directories exist. If the directory
    already exists, do nothing.
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)


def move_to_device(data, device: Union[str, device]):
    assert (
        isinstance(data, list)
        or isinstance(data, tuple)
        or isinstance(data, Tensor)
        or isinstance(data, dict)
    )
    if isinstance(data, list) or isinstance(data, tuple):
        return [d.to(device) for d in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        return data.to(device)


def save_checkpoint(
    model: Architecture,
    optimizer: Optimizer,
    path: str,
    global_step: Optional[int] = None,
    tensorboard_dir: Optional[str] = None,
    save_checkpoint_info_callback: Optional[Callable[[], dict]] = None,
):
    additional_info = (
        save_checkpoint_info_callback()
        if save_checkpoint_info_callback is not None
        else {}
    )
    info = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.kwargs,
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "tensorboard_dir": tensorboard_dir,
    }
    info.update(additional_info)
    torch.save(
        info,
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
    save_checkpoint_info_callback: Optional[Callable[[], dict]] = None,
    write_scaled_loss: bool = False,
) -> float:
    optimizer.zero_grad()
    batch_size = y.size(0)
    output = model(*x if isinstance(x, list) or isinstance(x, tuple) else x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if writer is not None:
        writer.add_scalar(
            "Train/Loss",
            loss.item() / batch_size if write_scaled_loss else loss.item(),
            global_step,
        )
    if (
        checkpoint_path is not None
        and save_every is not None
        and global_step % save_every == 0
    ):
        save_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            global_step,
            writer.get_logdir(),
            save_checkpoint_info_callback,
        )
    return loss.item()


def test_step(
    model: Architecture,
    loss_fn: Union[Callable, Module],
    x,
    y,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    write_scaled_loss: bool = False,
) -> float:
    with torch.no_grad():
        batch_size = y.size(0)
        output = model(*x if isinstance(x, list) or isinstance(x, tuple) else x)
        loss = loss_fn(output, y)
        if writer is not None:
            writer.add_scalar(
                "Test/Loss",
                loss.item() / batch_size if write_scaled_loss else loss.item(),
                global_step,
            )
    return loss.item()


# pytest considers any function starting with test_ as a test function
# however, we don't want to run this function as a test
# so we set __test__ to False
test_step.__test__ = False


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
    save_checkpoint_info_callback: Optional[Callable[[], dict]] = None,
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
    @param checkpoint_path: The path to save checkpoints to, such as "/path/to/checkpoint.pth"
    @param save_every: Save a checkpoint every n epochs
    @param test_every: Test the model every n epochs
    @param device: The device to train on
    @param save_checkpoint_info_callback: A callback function that returns a dictionary
        which will also be saved in the checkpoint file. This can be used to save additional
        information such as those needed to resume training from a checkpoint such
        as what dataset was used, the model type, etc.

    It is assumed that the dataloaders used returns tuples or lists of tensors,
    and that the last element of the tuple or list is the target used in loss function.
    It also assumes that model has already been moved to the specified device
    """
    # we first set epoch to seen_epochs in the event that training is
    # interrupted before the for loop starts, so we still have
    # the epoch variable available to use in save_checkpoint
    ensure_dir_exists(checkpoint_path)
    epoch = seen_epochs
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
    model.train()
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
            if (isinstance(x, list) or isinstance(x, tuple)) and len(x) == 1:
                x = x[0]
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
                if (isinstance(x, list) or isinstance(x, tuple)) and len(x) == 1:
                    x = x[0]
                test_loss = test_step(model, loss_fn, x, y, writer, epoch)
    except KeyboardInterrupt:
        save_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            epoch,
            writer.get_logdir(),
            save_checkpoint_info_callback,
        )
    finally:
        save_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            epoch,
            writer.get_logdir(),
            save_checkpoint_info_callback,
        )
