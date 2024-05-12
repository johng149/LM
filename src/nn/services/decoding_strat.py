from torch import Tensor
from src.nn.models.decoding_strat_model import AutoregressiveStrategy


class GreedyAutoregressiveStrategy(AutoregressiveStrategy):
    def __init__(self, info, device):
        super().__init__(info, device)

    def decode(self, output: Tensor):
        batch_size, vocab_size = output.shape
        greedy = output.argmax(dim=-1)
        return greedy.reshape(batch_size, -1)
