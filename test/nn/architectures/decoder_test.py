import pytest
import torch
from src.nn.architectures.decoder import Decoder
from src.nn.services.decoding_strat import GreedyAutoregressiveStrategy
from unittest.mock import MagicMock, patch


@pytest.fixture(params=[-1, 0, 1, 2])
def decoder(request):
    embed_dim = 128
    num_heads = 8
    factor = request.param
    vocab_size = 3
    max_len = 100
    num_layers = 2
    dropout = 0.1
    return (
        Decoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            factor=factor,
            vocab_size=vocab_size,
            max_len=max_len,
            num_layers=num_layers,
            dropout=dropout,
        ),
        factor,
    )


def test_init_kwargs(decoder):
    decoder, factor = decoder
    expected_kwargs = {
        "embed_dim": 128,
        "num_heads": 8,
        "factor": factor,
        "vocab_size": 3,
        "max_len": 100,
        "num_layers": 2,
        "dropout": 0.1,
    }
    assert decoder.init_kwargs() == expected_kwargs


def test_forward(decoder):
    decoder, _ = decoder
    kwargs = decoder.init_kwargs()
    vocab_size = kwargs["vocab_size"]
    max_len = kwargs["max_len"]
    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, max_len))
    output = decoder(x)
    assert output.shape == (batch_size, max_len, vocab_size)


def test_forward_too_long_sequence(decoder):
    decoder, _ = decoder
    kwargs = decoder.init_kwargs()
    vocab_size = kwargs["vocab_size"]
    max_len = kwargs["max_len"]
    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, max_len + 1))
    with pytest.raises(IndexError):
        decoder(x)


def test_forward_shortest_sequence(decoder):
    decoder, _ = decoder
    kwargs = decoder.init_kwargs()
    vocab_size = kwargs["vocab_size"]
    max_len = kwargs["max_len"]
    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, 1))
    output = decoder(x)
    assert output.shape == (batch_size, 1, vocab_size)


def test_forward_masked(decoder):
    decoder, _ = decoder
    kwargs = decoder.init_kwargs()
    vocab_size = kwargs["vocab_size"]
    max_len = kwargs["max_len"]
    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, max_len))
    mask = torch.randint(0, 2, (batch_size, max_len, max_len)).bool()
    mask = mask.unsqueeze(1)
    output = decoder(x, mask=mask)
    assert output.shape == (batch_size, max_len, vocab_size)


def test_naive_inference(decoder):
    with patch.object(Decoder, "forward") as mock_forward:
        decoder, _ = decoder
        kwargs = decoder.init_kwargs()
        vocab_size = kwargs["vocab_size"]
        max_len = kwargs["max_len"]
        batch_size = 1
        x = torch.randint(0, vocab_size, (batch_size, 2))
        mock_forward.return_value = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.4, 0.3],
                ]
            ]
        )
        tokenizer_info = MagicMock()
        tokenizer_info.pad_idx = -1
        tokenizer_info.eos_idx = 0
        device = torch.device("cpu")
        strat = GreedyAutoregressiveStrategy(tokenizer_info, device)
        output = decoder.naive_inference(x=x, strat=strat, max_len=1)
        assert output[0, -1] == 1
