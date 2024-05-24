from src.nn.architectures.encoder_decoder import EncoderDecoder
import torch
from src.datasets.utils.masking import (
    causal_self_attn_mask,
    self_attn_pad_mask,
    cross_attn_pad_mask,
    combine_masks_before_flip,
    process_tokens,
)


def test_encoder_decoder():
    emb_dim = 8
    num_heads = 2
    factor = -1
    vocab_size = 7
    max_len_enc = 10
    max_len_dec = 10
    num_enc_layers = 2
    num_dec_layers = 2
    dropout = 0.1
    model = EncoderDecoder(
        emb_dim,
        num_heads,
        factor,
        vocab_size,
        max_len_enc,
        max_len_dec,
        num_enc_layers,
        num_dec_layers,
        dropout,
    )
    pad_idx = 0
    encoder_input = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 3, 0, 0]])
    decoder_input = torch.tensor([[1, 2, 4, 0, 0], [1, 2, 3, 1, 0]])

    len_enc_not_pad, is_enc_not_pad = process_tokens(encoder_input, pad_idx)
    len_dec_not_pad, is_dec_not_pad = process_tokens(decoder_input, pad_idx)

    enc_pad_mask = self_attn_pad_mask(is_enc_not_pad)

    dec_pad_mask = self_attn_pad_mask(is_dec_not_pad)
    dec_causal_mask = causal_self_attn_mask(decoder_input)
    dec_mask = combine_masks_before_flip(dec_causal_mask, dec_pad_mask)

    enc_kv_dec_q_mask = cross_attn_pad_mask(is_enc_not_pad, is_dec_not_pad)

    output = model(
        encoder_input, enc_pad_mask, decoder_input, dec_mask, enc_kv_dec_q_mask
    )

    assert output.shape == (2, 5, 7)
