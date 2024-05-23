from transformers import AutoTokenizer
from src.tokenizers.models.info import Info


def tokenizer() -> Info:
    name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = AutoTokenizer.from_pretrained(name)

    # tokenizer doesn't have bos nor mask token,
    # so we add it
    bos_token_id = tokenizer.vocab_size
    mask_token_id = tokenizer.vocab_size + 1
    vocab_size = tokenizer.vocab_size + 2

    return Info(
        tokenizer=tokenizer,
        tokenizer_name=name,
        pad_idx=tokenizer.pad_token_id,
        bos_idx=bos_token_id,
        eos_idx=tokenizer.eos_token_id,
        mask_idx=mask_token_id,
        vocab_size=vocab_size,
    )
