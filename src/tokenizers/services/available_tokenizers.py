from src.tokenizers.services.gpt2 import tokenizer as gpt2_tokenizer
from src.tokenizers.services.helsinki_en_zh import tokenizer as helsinki_en_zh_tokenizer

available_tokenizers = {
    "gpt2": gpt2_tokenizer,
    "helsinki_en_zh": helsinki_en_zh_tokenizer,
}
