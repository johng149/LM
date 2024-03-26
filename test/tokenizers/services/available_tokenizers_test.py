from src.tokenizers.services.available_tokenizers import available_tokenizers


def test_available_tokenizers():
    avail = available_tokenizers
    assert len(avail) == 1
