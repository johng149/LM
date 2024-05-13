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
from torch.nn import Module
