from src.tokenizers.services.available_tokenizers import available_tokenizers
from src.tokenizers.models.info import Info


def test_available_tokenizers():
    avail = available_tokenizers
    assert len(avail) == 1


def test_gpt2_tokenizer():
    avail = available_tokenizers
    gpt2_info = avail["gpt2"]
    assert isinstance(gpt2_info, Info)
    assert gpt2_info.pad_idx == 50257
    assert gpt2_info.bos_idx == 50256
    assert gpt2_info.eos_idx == 50256
    assert gpt2_info.mask_idx == 50258
    assert gpt2_info.vocab_size == 50259
