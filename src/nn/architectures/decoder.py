import torch
from src.nn.models.decoding_strat_model import AutoregressiveStrategy
from src.nn.base.transformer import TransformerBlock
from src.nn.base.embedding import StableEmbedding
from src.nn.base.architecture import Architecture
from torch import Tensor
from typing import List, Tuple
from src.common.models.verification import Verification
from src.common.models.args_info import ArgInfo
from src.common.models.param_level import ParamLevel


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

    def verify_init_kwargs(self, **kwargs) -> Tuple[List[Verification], bool]:
        arg_info = {
            "embed_dim": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The dimension of the embedding",
                type=int,
            )
        }

    def init_kwargs(self) -> dict:
        return self.kwargs

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_embedding(torch.arange(seq_len).to(x.device))
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return self.classifier(x)

    def naive_inference(
        self, x: Tensor, strat: AutoregressiveStrategy, max_len: int
    ) -> Tensor:
        super().naive_inference(x, strat, max_len)
        batch_size, seq_len = x.shape
        for i in range(max_len):
            mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
            # since model has limited positional encoding, we take
            # a slice of the input tensor
            x_slice = x[:, -self.max_len :]
            logits = self.forward(x_slice, mask)
            last_logits = logits[:, -1, :]
            next_token = strat.decode(last_logits)
            x = torch.cat([x, next_token], dim=-1)
        return x
