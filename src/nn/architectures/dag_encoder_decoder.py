import torch
from src.nn.models.decoding_strat_model import DAGStrategy
from src.nn.base.transformer import TransformerBlock
from src.nn.base.embedding import StableEmbedding
from src.nn.base.architecture import Architecture
from src.nn.base.cross import XBlock
from src.nn.base.dag_out import OutputDAG
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
    self_attn_pad_mask,
    process_tokens,
    cross_attn_pad_mask,
    combine_masks,
)


class EncoderDecoderDAG(Architecture):
    def __init__(
        self,
        embed_dim,
        num_heads,
        factor,
        vocab_size,
        max_len_enc,
        max_len_dec,
        num_enc_layers,
        num_dec_layers,
        dag_heads,
        dag_phm_factor,
        dag_lm_head_factor,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.vocab_size = vocab_size
        self.max_len_enc = max_len_enc
        self.max_len_dec = max_len_dec
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dag_heads = dag_heads
        self.dag_phm_factor = dag_phm_factor
        self.dag_lm_head_factor = dag_lm_head_factor
        self.dropout_prob = dropout
        self.kwargs = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "factor": factor,
            "vocab_size": vocab_size,
            "max_len_enc": max_len_enc,
            "max_len_dec": max_len_dec,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "dag_heads": dag_heads,
            "dag_phm_factor": dag_phm_factor,
            "dag_lm_head_factor": dag_lm_head_factor,
            "dropout": dropout,
        }
        self.embedding = StableEmbedding(vocab_size, embed_dim)
        self.pos_embedding_enc = StableEmbedding(max_len_enc, embed_dim)
        self.pos_embedding_dec = StableEmbedding(max_len_dec, embed_dim)
        enc_layers = []
        for _ in range(num_enc_layers):
            enc_layers.append(TransformerBlock(embed_dim, num_heads, factor, dropout))
        self.enc_layers = torch.nn.ModuleList(enc_layers)
        dec_layers = []
        for _ in range(num_dec_layers):
            dec_layers.append(TransformerBlock(embed_dim, num_heads, factor, dropout))
            dec_layers.append(XBlock(embed_dim, num_heads, factor, dropout))
        self.dec_layers = torch.nn.ModuleList(dec_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.output_dag = OutputDAG(
            embed_dim, vocab_size, dag_heads, dag_phm_factor, dag_lm_head_factor
        )

        # weight tying
        self.embedding.embedding.weight = self.output_dag.lm_head.weight

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
            "max_len_enc": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The maximum length of the encoder input",
                type=int,
            ),
            "max_len_dec": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The maximum length of the decoder input",
                type=int,
            ),
            "num_enc_layers": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The number of layers in the encoder",
                type=int,
            ),
            "num_dec_layers": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The number of layers in the decoder",
                type=int,
            ),
            "dag_heads": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The number of DAG heads",
                type=int,
            ),
            "dag_phm_factor": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The factor for the DAG phm",
                type=int,
            ),
            "dag_lm_head_factor": ArgInfo(
                level=ParamLevel.REQUIRED,
                description="The factor for the DAG lm head",
                type=int,
            ),
            "dropout": ArgInfo(
                level=ParamLevel.OPTIONAL,
                description="The dropout probability",
                type=float,
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
            ("max_len_enc"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda max_len_enc: max_len_enc > 0,
                failure_msg="Max length of the encoder input must be greater than 0",
            ),
            ("max_len_dec"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda max_len_dec: max_len_dec > 0,
                failure_msg="Max length of the decoder input must be greater than 0",
            ),
            ("num_enc_layers"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda num_enc_layers: num_enc_layers > 0,
                failure_msg="Number of encoder layers must be greater than 0",
            ),
            ("num_dec_layers"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda num_dec_layers: num_dec_layers > 0,
                failure_msg="Number of decoder layers must be greater than 0",
            ),
            ("dag_heads"): ArgRelation(
                level=ParamLevel.REQUIRED,
                relation=lambda dag_heads: dag_heads > 0,
                failure_msg="Number of DAG heads must be greater than 0",
            ),
            ("dropout"): ArgRelation(
                level=ParamLevel.OPTIONAL,
                relation=lambda dropout: 0 <= dropout < 1,
                failure_msg="Dropout probability must be between 0 and 1",
            ),
        }
        return verify_args(arg_info, kwargs), verify_arg_relations(
            arg_relations, kwargs
        )

    def init_kwargs(self) -> dict:
        return self.kwargs

    def forward_encoder(self, encoder_input, enc_pad_mask):
        batch_size, seq_len_enc = encoder_input.shape
        enc_pad_mask = multiheadify([enc_pad_mask], self.num_heads)[0]

        encoder_input = self.embedding(encoder_input)
        encoder_input += self.pos_embedding_enc(
            torch.arange(seq_len_enc, device=encoder_input.device)
        )
        for layer in self.enc_layers:
            encoder_input = layer(encoder_input, enc_pad_mask)
        return encoder_input

    def forward_decoder(
        self, encoder_input, decoder_input, dec_mask, enc_kv_dec_q_mask, vertex_lens
    ):
        batch_size, seq_len_dec = decoder_input.shape
        dec_mask, enc_kv_dec_q_mask = multiheadify(
            [dec_mask, enc_kv_dec_q_mask], self.num_heads
        )

        # here we don't need the self.embedding layer, just
        # the pos_embedding_dec layer.
        # this is because for a DAG model, the decoder input
        # are the indices for vertices in the graph,
        # and so the embedding layer should be just the
        # positional embedding for those vertices
        # no vocabulary embeddings are needed
        decoder_input = self.pos_embedding_dec(decoder_input)

        for layer in self.dec_layers:
            if isinstance(layer, TransformerBlock):
                decoder_input = layer(decoder_input, dec_mask)
            else:
                decoder_input = layer(decoder_input, encoder_input, enc_kv_dec_q_mask)

        x = self.dropout(decoder_input)
        out = self.output_dag(x, vertex_lens)
        transition, emission = out
        return transition, emission

    def forward(
        self,
        encoder_input,
        enc_pad_mask,
        decoder_input,
        dec_mask,
        env_kv_dec_q_mask,
        vertex_lens,
    ):
        encoder_output = self.forward_encoder(encoder_input, enc_pad_mask)
        return self.forward_decoder(
            encoder_output, decoder_input, dec_mask, env_kv_dec_q_mask, vertex_lens
        )

    def naive_inference(
        self,
        strat: DAGStrategy,
        encoder_input: Tensor,
        coefficient: int,
        *args,
    ):
        batch_size, seq_len_enc = encoder_input.shape
        assert batch_size == 1
        assert (encoder_input == strat.pad_id()).sum() == 0
        bos_idx = strat.bos_id()
        eos_idx = strat.eos_id()
        dec_input_length = min(self.max_len_dec, seq_len_enc * coefficient)
        dec_input = torch.arange(0, dec_input_length, device=encoder_input.device)
        dec_input = dec_input.unsqueeze(0)

        len_enc_not_pad, is_enc_not_pad = process_tokens(encoder_input, strat.pad_id())
        enc_pad_mask = self_attn_pad_mask(is_enc_not_pad)

        enc_output = self.forward_encoder(encoder_input, enc_pad_mask)

        # for decoder we process token with hard-coded pad_idx
        # of -1, since we don't need to mask out any tokens
        len_dec_not_pad, is_dec_not_pad = process_tokens(dec_input, -1)
        dec_mask = self_attn_pad_mask(is_dec_not_pad)

        enc_kv_dec_q_mask = cross_attn_pad_mask(is_enc_not_pad, is_dec_not_pad)

        transition, emission = self.forward_decoder(
            enc_output,
            dec_input,
            dec_mask,
            enc_kv_dec_q_mask,
        )

        return strat.decode(transition, emission)
