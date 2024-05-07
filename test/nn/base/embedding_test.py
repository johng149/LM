from src.nn.base.embedding import StableEmbedding
import torch


def test_embedding():
    batch_size = 2
    seq_len = 3
    vocab_size = 5
    emb_dim = 4

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    emb = StableEmbedding(vocab_size, emb_dim)
    out = emb(tokens)
    assert out.shape == (batch_size, seq_len, emb_dim)
