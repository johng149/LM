from src.tokenizers.models.info import Info
from torch import device
from torch import Tensor
from typing import Union


class DecodingStrategy:
    def __init__(self, info: Info, device: Union[device, str]):
        self.info = info
        self.device = device

    def pad_id(self) -> int:
        return self.info.pad_idx

    def eos_id(self) -> int:
        return self.info.eos_idx

    def bos_id(self) -> int:
        return self.info.bos_idx

    def decode(self, output: Tensor) -> Tensor:
        raise NotImplementedError()


class AutoregressiveStrategy(DecodingStrategy):
    def __init__(self, info: Info, device: Union[device, str]):
        super().__init__(info, device)

    def decode(self, output: Tensor) -> Tensor:
        """
        This is more specific than a generic decoding strategy,
        it must be that the output tensor's batch dimension
        is equal to the output tensor's batch dimension.

        @param output: the output tensor, shape (batch_size, vocab_size)
        @return: the decoded tensor, shape (batch_size,)
        """
        raise NotImplementedError()


class DAGStrategy(DecodingStrategy):
    def __init__(self, info: Info, device: Union[device, str]):
        super().__init__(info, device)
