from transformers import PreTrainedTokenizerBase


class Info:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_name: str,
        pad_idx: int,
        bos_idx: int,
        eos_idx: int,
        mask_idx: int,
        vocab_size: int,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.mask_idx = mask_idx
        self.vocab_size = vocab_size
