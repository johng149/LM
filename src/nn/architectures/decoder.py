import torch
from src.nn.models.decoding_strat_model import AutoregressiveStrategy
from src.nn.base.transformer import TransformerBlock
from src.nn.base.embedding import StableEmbedding
from src.nn.base.architecture import Architecture
from torch import Tensor
from typing import List, Tuple
from src.common.models.verification import Verification
from src.common.models.args_info import ArgInfo
from src.common.models.args_relation import ArgRelation
from src.common.models.param_level import ParamLevel
from src.common.services.verification import verify_args, verify_arg_relations
from src.datasets.utils.masking import (
    causal_self_attn_mask,
    multiheadify,
)


class Decoder(Architecture):
    def __init__(
        self, embed_dim, num_heads, factor, vocab_size, max_len, num_layers, dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.kwargs = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "factor": factor,
            "vocab_size": vocab_size,
            "max_len": max_len,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.embedding = StableEmbedding(vocab_size, embed_dim)
        self.pos_embedding = StableEmbedding(max_len, embed_dim)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, factor, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.classifier = torch.nn.Linear(embed_dim, vocab_size)

        # weight tying
        self.embedding.embedding.weight = self.classifier.weight

    def verify_init_kwargs_helper(
        self, **kwargs
    ) -> Tuple[List[Verification], bool, List[Verification], bool]:
        arg_info = {
            "embed_dim": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The dimension of the embedding",
                type=int,
            ),
            "num_heads": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The number of heads in the multihead attention",
                type=int,
            ),
            "factor": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The factor for the dimension of the feedforward network",
                type=int,
            ),
            "vocab_size": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The size of the vocabulary",
                type=int,
            ),
            "max_len": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The maximum length of the input sequence",
                type=int,
            ),
            "num_layers": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The number of layers in the transformer",
                type=int,
            ),
            "dropout": ArgInfo(
                level=ParamLevel.OPTIONAL,
                description="The dropout probability",
                type=float,
                default=0.1,
            ),
        }
        arg_relations = {
            ("embed_dim", "num_heads"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda embed_dim, num_heads: embed_dim % num_heads == 0,
                failure_msg="Embedding dimension must be divisible by the number of heads",
            ),
            ("vocab_size"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda vocab_size: vocab_size > 0,
                failure_msg="Vocabulary size must be greater than 0",
            ),
            ("max_len"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda max_len: max_len > 0,
                failure_msg="Max length must be greater than 0",
            ),
            ("num_layers"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda num_layers: num_layers > 0,
                failure_msg="Number of layers must be greater than 0",
            ),
            ("dropout"): ArgRelation(
                level=ParamLevel.OPTIONAL,
                relation=lambda dropout: 0 <= dropout < 1,
                failure_msg="Dropout probability must be between 0 and 1",
            ),
        }
        v1, e1 = verify_args(arg_info, **kwargs)
        v2, e2 = verify_arg_relations(arg_relations, **kwargs)
        return v1, e1, v2, e2

    def init_kwargs(self) -> dict:
        return self.kwargs

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        mask = multiheadify([mask], self.num_heads)[0]
        x = self.embedding(x) + self.pos_embedding(torch.arange(seq_len).to(x.device))
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)

    def naive_inference(
        self, x: Tensor, strat: AutoregressiveStrategy, max_len: int
    ) -> Tensor:
        super().naive_inference(x, strat, max_len)
        eos_idx = strat.info.eos_idx
        for i in range(max_len):
            # since model has limited positional encoding, we take
            # a slice of the input tensor
            x_slice = x[:, -self.max_len :]
            _, x_len = x_slice.shape
            mask = causal_self_attn_mask(x_slice)
            mask = multiheadify([mask], self.num_heads)[0]
            logits = self.forward(x_slice, mask)
            last_logits = logits[:, -1, :]
            next_token = strat.decode(last_logits)
            x = torch.cat([x, next_token], dim=-1)
            if (next_token == eos_idx).all():
                break
        return x
