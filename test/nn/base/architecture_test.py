from src.nn.base.architecture import Architecture
import pytest


def test_architecture():
    arch = Architecture()
    with pytest.raises(NotImplementedError):
        arch.init_kwargs()
