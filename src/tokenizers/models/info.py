from typing import TypedDict
from transformers import PreTrainedTokenizerBase

Info = TypedDict(
    'Info',
    {
        "tokenizer": PreTrainedTokenizerBase,
        "pad_idx": int,
        "bos_idx": int,
        "eos_idx": int,
        "mask_idx": int,
        "vocab_size": int,
    }
)