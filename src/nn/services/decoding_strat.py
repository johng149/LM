from torch import Tensor
from src.nn.models.decoding_strat_model import AutoregressiveStrategy
from src.nn.models.decoding_strat_model import DAGStrategy
import torch


class GreedyAutoregressiveStrategy(AutoregressiveStrategy):
    def __init__(self, info, device):
        super().__init__(info, device)

    def decode(self, output: Tensor):
        batch_size, vocab_size = output.shape
        greedy = output.argmax(dim=-1)
        return greedy.reshape(batch_size, -1)


class GreedyDAGStrategy(DAGStrategy):
    def __init__(self, info=None, device=None):
        self.info = info  # we don't actually need the info for this strategy
        self.device = device

    def decode_single(self, transition: Tensor, emissions: Tensor):
        vertex_count, vocab_size = emissions.shape

        tokens = torch.argmax(emissions, dim=1)
        edges = torch.argmax(transition, dim=1)

        edges[edges == 0] = vertex_count

        i = 0
        output = [tokens[i].item()]
        while i < vertex_count:
            i = edges[i].item()
            if i >= vertex_count:
                break
            output.append(tokens[i].item())

        return torch.tensor(
            output, device=self.device if self.device is not None else emissions.device
        )

    def decode(self, transition: Tensor, emissions: Tensor):
        batch_size, vertex_count, vocab_size = emissions.shape

        # unfortunately, DAG decoding doesn't support parallelizing
        # across the batch dimension so we'll need a loop
        decoded = []

        for i in range(batch_size):
            decoded.append(self.decode_single(transition[i], emissions[i]))
        return torch.stack(decoded)
