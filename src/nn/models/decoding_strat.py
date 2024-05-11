from src.tokenizers.models.info import Info
from torch import device
from torch import Tensor


class DecodingStrategy:
    def __init__(self, info: Info, device: device):
        self.info = info
        self.device = device

    def pad_id(self) -> int:
        return self.info.pad_idx

    def eos_id(self) -> int:
        return self.info.eos_idx

    def decode(self, output: Tensor) -> Tensor:
        raise NotImplementedError()


class AutoregressiveStrategy(DecodingStrategy):
    def __init__(self, info: Info, device: device):
        super().__init__(info, device)


class DAGStrategy(DecodingStrategy):
    def __init__(self, info: Info, device: device):
        super().__init__(info, device)
