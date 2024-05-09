import torch
from src.nn.base.transformer import TransformerBlock
from src.nn.base.embedding import StableEmbedding
from src.nn.base.architecture import Architecture


class Decoder(Architecture):
    def __init__(self, embed_dim, num_heads, factor, vocab_size, max_len, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.kwargs = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "factor": factor,
            "vocab_size": vocab_size,
            "max_len": max_len,
            "num_layers": num_layers,
        }
        self.embedding = StableEmbedding(vocab_size, embed_dim)
        self.pos_embedding = StableEmbedding(max_len, embed_dim)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, factor) for _ in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.classifier = torch.nn.Linear(embed_dim, vocab_size)

    def init_kwargs(self) -> dict:
        return self.kwargs

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_embedding(torch.arange(seq_len).to(x.device))
        print(x.shape)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return self.classifier(x)
