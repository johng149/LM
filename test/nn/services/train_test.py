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
        model, optimizer, checkpoint_path, global_step
    )
