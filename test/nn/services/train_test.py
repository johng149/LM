import pytest
import torch
from src.nn.services.train import (
    move_to_device,
    save_checkpoint,
    train_step,
    test_step,
    train,
)
from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.nn import Module, Parameter
from unittest.mock import MagicMock, patch
from pyfakefs.fake_filesystem import FakeFilesystem
import os
from unittest import mock


class MockArchitecture(Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kwargs = {
            "input_shape": input_shape,
            "output_shape": output_shape,
            **kwargs,
        }
        self.linear = torch.nn.Linear(input_shape, output_shape)

    def init_kwargs(self):
        return self.kwargs

    def forward(self, x):
        return self.linear(x)


def test_move_to_device():
    device = "mock_device"
    data = [
        MagicMock(to=MagicMock(return_value="mock_device")),
        MagicMock(to=MagicMock(return_value="mock_device")),
    ]
    moved = move_to_device(data, device)
    assert all([m == "mock_device" for m in moved])


@patch("src.nn.services.train.save_checkpoint")
def test_train_step(mock_save_checkpoint):
    input_shape = 10
    output_shape = 1
    model = MockArchitecture(input_shape, output_shape)
    optimizer = SGD(model.parameters(), lr=0.1)
    loss_fn = MSELoss()
    x = torch.randn(10, input_shape)
    y = torch.randn(10, output_shape)
    with torch.no_grad():
        expected_loss: float = loss_fn(model(x), y).item()
    writer = MagicMock()
    writer.get_logdir.return_value = "mock_logdir"
    global_step = 3
    checkpoint_path = "mock_checkpoint_path/model.pt"
    save_every = 3
    loss: float = train_step(
        model,
        optimizer,
        loss_fn,
        x,
        y,
        writer=writer,
        global_step=global_step,
        checkpoint_path=checkpoint_path,
        save_every=save_every,
    )
    eps = 1e-6
    assert abs(loss - expected_loss) < eps
    assert mock_save_checkpoint.call_count == 1
    assert mock_save_checkpoint.call_args == mock.call(
        model, optimizer, checkpoint_path, global_step, writer.get_logdir(), None
    )
    assert writer.add_scalar.call_count == 1
    assert writer.add_scalar.call_args == mock.call(
        "Train/Loss", expected_loss, global_step
    )


def test_test_step():
    input_shape = 10
    output_shape = 1
    model = MockArchitecture(input_shape, output_shape)
    loss_fn = MSELoss()
    x = torch.randn(10, input_shape)
    y = torch.randn(10, output_shape)
    with torch.no_grad():
        expected_loss: float = loss_fn(model(x), y).item()
    writer = MagicMock()
    global_step = 3
    loss: float = test_step(
        model,
        loss_fn,
        x,
        y,
        writer=writer,
        global_step=global_step,
    )
    eps = 1e-6
    assert abs(loss - expected_loss) < eps
    assert writer.add_scalar.call_count == 1
    assert writer.add_scalar.call_args == mock.call(
        "Test/Loss", expected_loss, global_step
    )


@patch("src.nn.services.train.save_checkpoint")
@patch("src.nn.services.train.move_to_device", wraps=move_to_device)
@patch("src.nn.services.train.ensure_dir_exists")
def test_train(
    mock_ensure_dir_exists,
    mock_move_to_device,
    mock_save_checkpoint,
):
    input_shape = 10
    output_shape = 1
    model = MockArchitecture(input_shape, output_shape)
    optimizer = SGD(model.parameters(), lr=0.1)
    loss_fn = MSELoss()
    seen_epochs = 0
    target_epochs = 6
    train_x, train_y = torch.randn(10, input_shape), torch.randn(10, output_shape)
    train_loader = DataLoader(list(zip(train_x, train_y)), batch_size=10)
    test_x, test_y = torch.randn(10, input_shape), torch.randn(10, output_shape)
    test_loader = DataLoader(list(zip(test_x, test_y)), batch_size=10)
    writer = MagicMock()
    checkpoint_path = "mock_checkpoint_path/model.pt"
    save_every = 3
    test_every = 2
    device = "cpu"
    train(
        model,
        optimizer,
        loss_fn,
        seen_epochs,
        target_epochs,
        train_loader,
        test_loader,
        writer,
        checkpoint_path,
        save_every,
        test_every,
        device,
    )
    assert mock_ensure_dir_exists.call_count == 1
    assert mock_ensure_dir_exists.call_args == mock.call(checkpoint_path)
    assert mock_move_to_device.call_count == 9
    # 3 save checkpoints, 2 from the loop, 1 from
    # the finally block
    assert mock_save_checkpoint.call_count == 3
    assert mock_save_checkpoint.call_args == mock.call(
        model, optimizer, checkpoint_path, 5, writer.get_logdir(), None
    )
