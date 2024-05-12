from src.nn.services.decoding_strat import GreedyAutoregressiveStrategy
import torch


def test_greedy():
    strat = GreedyAutoregressiveStrategy({}, torch.device("cpu"))
    output = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    assert torch.equal(strat.decode(output), torch.tensor([2, 1]).reshape(2, -1))
