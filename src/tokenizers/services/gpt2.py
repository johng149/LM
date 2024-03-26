from transformers import AutoTokenizer
from src.tokenizers.models.info import Info


def tokenizer() -> Info:
    name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(name)

    # this tokenizer doesn't have pad token, so we add it
    # same with mask token
    pad_token_id = tokenizer.vocab_size
    mask_token_id = tokenizer.vocab_size + 2
    vocab_size = tokenizer.vocab_size + 2
    return {
        "tokenizer": tokenizer,
        "pad_token_id": pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "mask_token_id": mask_token_id,
        "vocab_size": vocab_size,
    }
