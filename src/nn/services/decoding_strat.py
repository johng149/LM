from torch import Tensor
from src.nn.models.decoding_strat import AutoregressiveStrategy


class GreedyAutoregressiveStrategy(AutoregressiveStrategy):
    def __init__(self, info, device):
        super().__init__(info, device)

    def decode(self, output: Tensor):
        return output.argmax(dim=-1)
