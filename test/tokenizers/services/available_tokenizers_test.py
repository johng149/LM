from src.tokenizers.services.available_tokenizers import available_tokenizers
from src.tokenizers.models.info import Info
import inspect


def test_available_tokenizers():
    avail = available_tokenizers
    assert len(avail) == 1


def test_available_tokenizer_types():
    for key in available_tokenizers:
        token_funct = available_tokenizers[key]
        assert callable(token_funct)  # check is callable, that is, a function
        assert not inspect.isclass(
            token_funct
        )  # check is not class, since classes can be callable
        assert not isinstance(
            token_funct, Info
        )  # objects can also be callable if they have __call__ method, but we only want functions


def test_gpt2_tokenizer():
    avail = available_tokenizers
    gpt2_info = avail["gpt2"]()
    assert isinstance(gpt2_info, Info)
    assert gpt2_info.pad_idx == 50257
    assert gpt2_info.bos_idx == 50256
    assert gpt2_info.eos_idx == 50256
    assert gpt2_info.mask_idx == 50258
    assert gpt2_info.vocab_size == 50259
